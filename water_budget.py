#!/usr/bin/env python

"""Water budget module

This module computes:

- the snow/rain fraction
- the snow cover
- the evapotranspiration
- the soil water availability
"""

__all__ = [
    "main_snow",
    "main_soil_water",
    "calc_snow",
    "calc_soil_water",
]

import dask.array as dask_arr
import numpy as np
import os
import xarray as xr
from typing import Iterable

import snowmaus


def calc_snow(ds: xr.Dataset) -> xr.Dataset:
    """Calculate snowfall and melt

    :param ds: Dataset containing variables "precipitation", "min_air_temp",
        "max_air_temp", and "initial_snowcover"
    :type ds: xr.Dataset
    :return: Dataset containing variables "snowfall", "meltwater_production",
        and "snowcover"
    :rtype: xr.Dataset
    """
    if ds.precipitation.isnull().all():
        template = xr.DataArray(
            dask_arr.zeros_like(ds.precipitation, dtype="f4", chunks=(-1, 37, 41)),
            dims=ds.precipitation.dims,
            coords=ds.precipitation.coords,
        )
        return xr.merge(
            [
                template.rename("snowfall"),
                template.rename("meltwater_production"),
                template.rename("snowcover"),
            ]
        )
    potential_melt = snowmaus.meltwater_production(
        ds.min_air_temp.values, ds.max_air_temp.values
    )
    snowfall = snowmaus.snowfall(ds.precipitation.values, ds.min_air_temp.values)
    melt = np.zeros_like(snowfall)
    snowcover = np.zeros_like(snowfall)
    for i in range(snowcover.shape[0]):
        snowcover[i] = snowcover[i - 1] if i != 0 else ds.initial_snowcover.values
        snowcover[i] -= snowmaus.sublimed_snowcover(snowcover[i])
        snowcover[i] += np.where(np.isnan(snowfall[i]), 0, snowfall[i])
        melt[i] = np.where(
            potential_melt[i] > snowcover[i], snowcover[i], potential_melt[i]
        )
        snowcover[i] -= np.where(np.isnan(melt[i]), 0, melt[i])
    snowcover = np.where(ds.min_air_temp.isnull().all("time"), np.nan, snowcover)
    melt = np.where(ds.min_air_temp.isnull().all("time"), np.nan, melt)
    return xr.Dataset(
        {
            "snowfall": (ds.precipitation.dims, snowfall),
            "meltwater_production": (ds.precipitation.dims, melt),
            "snowcover": (ds.precipitation.dims, snowcover),
        },
        coords=ds.precipitation.coords,
    )


def calc_soil_water(ds: xr.Dataset) -> xr.Dataset:
    """Calculate evapotranspiration and soil water depletion

    :param ds: Dataset containing variables "Kc_factor", "plant_height",
        "precipitation", "wind_speed", "rel_humidity" "pot_evapotransp",
        "snowfall", and "meltwater_production"
    :type ds: xr.Dataset
    :return: Dataset containing variables "evapotranspiration", "evapo_ETC", and
        "soil_depletion"
    :rtype: xr.Dataset
    """
    if ds.Kc_factor.isnull().all():
        template = xr.DataArray(
            dask_arr.zeros(
                shape=(*ds.Kc_factor.shape, 2), dtype="float", chunks=(4, -1, 37, 41, 2)
            ),
            dims=[*ds.Kc_factor.dims, "layer"],
            coords={
                **{k: ds.Kc_factor[k] for k in ds.Kc_factor.coords},
                "layer": ds.TAW.layer,
            },
        )
        return xr.merge(
            [
                template.rename("evapotranspiration"),
                template.rename("evapo_ETC"),
                template.rename("soil_depletion"),
            ]
        )
    pot_interc_precip = dask_arr.maximum(
        # activate one of the two implementations
        # # below implements the formula from the Schaumber Thesis
        # 1.875 * ds.Kc_factor - 0.25, 0.2 * ds.pot_evapotransp
        # below implements the (original) ARIS interception
        # ! assumes Kc is 1.2 between mid and end season
        # ! makes irrelevant mistake after end season (=0.13 instead 0.1)
        ds.Kc_factor
        / 3
    )
    liq_precip = ds.precipitation - ds.snowfall
    incoming_water = xr.where(
        pot_interc_precip <= liq_precip, liq_precip - pot_interc_precip, 0
    )
    incoming_water += ds.meltwater_production
    root_factor = xr.DataArray([0.6, 0.4], coords={"layer": ["top", "sub"]})
    ET0 = ds.pot_evapotransp * root_factor
    climEff = xr.where(
        ds.plant_height.isnull(),
        0,
        (0.04 * (ds.wind_speed - 2))
        - (0.004 * (ds.rel_humidity - 45)) * (ds.plant_height / 3) ** 0.3,
    )  # !! should probably be bounded
    Kc_plus_climEff = ds.Kc_factor + climEff
    ETC = ds.Kc_factor * ET0

    D_r = xr.DataArray(
        np.float32(0),
        dims=[*Kc_plus_climEff.dims, "layer"],
        coords={
            **{k: Kc_plus_climEff[k] for k in Kc_plus_climEff.coords},
            "layer": ds.TAW.layer,
        },
    ).rename("soil_depletion")
    ET = xr.DataArray(
        np.float32(np.nan),
        dims=[*Kc_plus_climEff.dims, "layer"],
        coords={
            **{k: Kc_plus_climEff[k] for k in Kc_plus_climEff.coords},
            "layer": ds.TAW.layer,
        },
    ).rename("evapotranspiration")
    for t in incoming_water.time[~incoming_water.time.dt.month.isin([1, 2, 12])].values:
        i = np.argwhere(t == incoming_water.time.values).flatten()[0]
        p__upper_lim, p__lower_lim = 0.1, 0.8
        # p_T generally depends on crop. however, values not available
        p_T = 0.6
        p = p_T + (0.04 * (5 - ETC.sel(time=t)))
        p = xr.where(
            p < p__lower_lim, p__lower_lim, xr.where(p > p__upper_lim, p__upper_lim, p)
        )
        Ks_i = (1 - D_r.isel(time=i - 1) / ds.TAW).values / (1 - p)
        Ks_i = xr.where(Ks_i < 0, 0, xr.where(Ks_i > 1, 1, Ks_i))
        ET[:, i] = Ks_i * ET0.sel(time=t) * Kc_plus_climEff.sel(time=t)
        toplayerimbalance = (
            incoming_water.sel(time=t)
            - D_r.isel(time=i - 1).squeeze().sel(layer="top")
            - ET.sel(time=t, layer="top")
        )
        maybe_new_top_layer_value = xr.where(
            toplayerimbalance < -ds.TAW.sel(layer="top"),
            ds.TAW.sel(layer="top"),
            -toplayerimbalance,
        )
        DP = xr.where(toplayerimbalance > 0, toplayerimbalance, 0)
        sublayerimbalance = (
            DP
            - D_r.isel(time=i - 1).squeeze().sel(layer="sub")
            - ET.sel(time=t, layer="sub")
        )
        maybe_new_sub_layer_value = xr.where(
            sublayerimbalance < -ds.TAW.sel(layer="sub"),
            ds.TAW.sel(layer="sub"),
            -sublayerimbalance,
        )
        potential_depletion = xr.concat(
            [maybe_new_top_layer_value, maybe_new_sub_layer_value], dim="layer"
        )
        D_r[:, i] = xr.where(potential_depletion < 0, 0, potential_depletion)
    return xr.merge([ET, ETC.rename("evapo_ETC"), D_r])


def main_soil_water(years: Iterable[int]):
    """Load input data and write soil related results to Zarr store

    :param years: List of years to compute
    :type years: Iterable[int]
    """
    for year in years:
        if os.path.isdir(f"../data/intermediate/{year}.zarr/soil_depletion"):
            print(f"! WARNING: {year}.zarr/soil_depletion already exists. Skipping.")
            continue
        print("Calculating soil water and evapotranspiration for year", year)
        pheno_ds = xr.open_zarr(
            f"../data/intermediate/{year}.zarr", decode_coords="all"
        )
        snow_ds = xr.open_zarr(
            f"../data/intermediate/snow_{year}.zarr", decode_coords="all"
        )
        meteo_ds = xr.open_zarr(f"../data/input/{year}.zarr", decode_coords="all")
        TAW = xr.open_dataarray("../data/input/soil_taw.nc", decode_coords="all")
        main_ds = xr.merge([pheno_ds, meteo_ds, TAW, snow_ds]).drop_vars(
            ["lambert_conformal_conic"]
        )
        template = xr.DataArray(
            dask_arr.zeros(
                shape=(*main_ds.Kc_factor.shape, 2),
                dtype="f4",
                chunks=(4, -1, 37, 41, 2),
            ),
            dims=[*main_ds.Kc_factor.dims, "layer"],
            coords={
                **{k: main_ds.Kc_factor[k] for k in main_ds.Kc_factor.coords},
                "layer": TAW.layer,
            },
        )
        D_r = main_ds.map_blocks(
            calc_soil_water,
            template=xr.merge(
                [
                    template.rename("evapotranspiration"),
                    template.rename("evapo_ETC"),
                    template.rename("soil_depletion"),
                ]
            ),
        )
        D_r.drop_encoding().to_zarr(f"../data/intermediate/{year}.zarr", mode="a-")


def main_snow(years: Iterable[int]):
    """Load input data and write snow related results to Zarr store

    :param years: List of years to compute
    :type years: Iterable[int]
    """
    for year in years:
        if os.path.isdir(f"../data/intermediate/snow_{year}.zarr"):
            print(f"! WARNING: snow_{year}.zarr already exists. Skipping.")
            continue
        main_ds = xr.open_zarr(f"../data/input/{year}.zarr", decode_coords="all")
        if os.path.isdir(
            f"../data/intermediate/snow_{year-1}.zarr"
        ) and "snowcover" in xr.open_zarr(f"../data/intermediate/snow_{year-1}.zarr"):
            main_ds["initial_snowcover"] = xr.open_zarr(
                f"../data/intermediate/snow_{year-1}.zarr"
            ).snowcover.isel(time=-1)
            # next step is necessary! somehow this `xr.where` changes how the data looks internally
            main_ds["precipitation"] = xr.where(
                main_ds.time.dt.month == 7, 0, main_ds.precipitation
            )
        else:
            print(
                "\n! WARNING: snowcover data for previous year are missing; initializing with "
                "zero snowcover\n"
                "consider not using data of this year for computing yield expectations\n"
            )
            main_ds["initial_snowcover"] = xr.zeros_like(
                main_ds.precipitation.isel(time=0)
            )
            main_ds["precipitation"] = xr.where(
                main_ds.time.dt.month < 8, 0, main_ds.precipitation
            )
        print("Calculating snow related variables for year", year)
        template = xr.DataArray(
            dask_arr.zeros_like(main_ds.precipitation, dtype="f4", chunks=(-1, 37, 41)),
            dims=main_ds.precipitation.dims,
            coords=main_ds.precipitation.coords,
        )
        main_ds.map_blocks(
            calc_snow,
            template=xr.merge(
                [
                    template.rename("snowfall"),
                    template.rename("meltwater_production"),
                    template.rename("snowcover"),
                ]
            ),
        ).drop_encoding().to_zarr(f"../data/intermediate/snow_{year}.zarr", mode="a")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="computes either the snow/melt or the soil water "
        "and evapotranspiration"
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["snow", "soil", "auto"],
        default="auto",
        help="choose which part of the water budget to compute",
    )
    parser.add_argument(
        "years",
        type=int,
        nargs="*",
        default=[2020, 2021, 2023],
        help="list years to compute",
    )
    parser.add_argument("--workers", type=int, default=4, help="number of dask workers")
    parser.add_argument(
        "--mem-per-worker",
        type=str,
        default="2Gb",
        help='memory per worker, e.g. "5.67Gb"',
    )
    args = parser.parse_args()
    args.years = sorted(args.years)

    if args.mode == "auto":
        if all(
            os.path.isdir(f"../data/intermediate/snow_{year}.zarr")
            for year in args.years
        ):
            print(
                "Snow related variables are present, assuming you mean to have the soil part "
                "of the water budget computed"
            )
            args.mode = "soil"
        else:
            print(
                "Snow related variables are missing for year(s):",
                ", ".join(
                    [
                        str(year)
                        for year in args.years
                        if not os.path.isdir(f"../data/intermediate/snow_{year}.zarr")
                    ]
                )
                + ".",
                "Computing these now.",
            )
            args.mode = "snow"

    from dask.distributed import LocalCluster, Client

    print(f"Starting dask ({args.workers} CPUs, each {args.mem_per_worker} RAM)")
    client = Client(
        LocalCluster(
            n_workers=args.workers, memory_limit=args.mem_per_worker, death_timeout=30
        )
    )
    print("... access the dashboard at", client.dashboard_link)

    try:
        if args.mode == "snow":
            main_snow(args.years)
        else:
            main_soil_water(args.years)
    except (FileNotFoundError,) as err:
        if str(err).startswith("Unable to find group"):
            print(
                "\n! ERROR: data missing. Verify that the necessary data are available.\n"
            )
            raise
    finally:
        client.close()
        print("Closed dask client\n")

    print(f"Sucessfully computed {args.mode} related variables!\n")
    if args.mode == "snow":
        print(
            "Continue by computing the crop coefficients (needed to calculate the "
            "evapotranspiration later) by running\n\t`python phenology.py [year1 ...]`\n"
        )
    else:
        print(
            "Continue by computing the expected yield by running\n\t`python "
            "yield_expectation.py [year1 ...]`\n"
        )
