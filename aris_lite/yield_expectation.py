#!/usr/bin/env python

"""Yield module

Estimates yield based on stress indices
"""

__all__ = [
    "main_combined_stress",
    "main_yield",
    "calc_combined_stress",
    "calc_yield",
]

import os
from typing import Iterable
import xarray as xr
import numpy as np


def set_combined_stress_meta(da: xr.DataArray) -> xr.DataArray:
    """Set metadata for the combined stress index"""
    return da.rename("combined_stress").assign_attrs(
        dict(
            unit="",
            long_name="daily crop specific stress index based on maximum "
            "surface air temperature and soil water saturation",
            description="Index of combination of plant growth inhibiting "
            "factors. Used for yield estimation.",
        )
    )


def calc_combined_stress(ds: xr.Dataset) -> xr.DataArray:
    """Calculate combined stress index

    The combined stress index folds the water/drought stress and the heat stress

    :param ds: Dataset containing variables "waterstress" and "max_air_temp"
    :type ds: xr.Dataset
    :return: Combined stress index
    :rtype: xr.DataArray
    """
    combined_stress = ds.waterstress.where(False)
    for i in range(combined_stress.shape[0]):
        if combined_stress[i].crop == "winter wheat":
            combined_stress[i] = xr.where(
                ds.max_air_temp > 26,
                ds.waterstress[i] * (ds.max_air_temp - 25),
                ds.waterstress[i],
            )
        if combined_stress[i].crop in ["spring barley", "maize"]:
            combined_stress[i] = xr.where(
                np.logical_and(ds.max_air_temp > 30, ds.waterstress[i] > 33),
                (ds.waterstress[i] * (ds.max_air_temp - 29)) - 33,
                ds.waterstress[i],
            ).where(ds.time.dt.month >= 3)
        if combined_stress[i].crop == "grassland":
            combined_stress[i] = ds.waterstress[i]
        if combined_stress[i].crop in ["winter wheat", "spring barley"]:
            combined_stress[i] = combined_stress[i].where(ds.time.dt.month >= 3)
        if combined_stress[i].crop in ["grassland", "maize"]:
            combined_stress[i] = combined_stress[i].where(ds.time.dt.month >= 5)
    combined_stress = combined_stress.where(
        (ds.Kc_factor > 0.5)[:, ::-1].cumsum("time") != 0
    )
    return set_combined_stress_meta(combined_stress)


def main_combined_stress(years: Iterable[int]):
    """Load input data and write combined stress to Zarr store

    :param years: List of years to compute
    :type years: Iterable[int]
    """
    TAW = xr.open_dataarray("../data/input/soil_taw.nc", decode_coords="all")
    for year in years:
        if os.path.isdir(f"../data/intermediate/CSI_{year}.zarr"):
            print(f"! WARNING: CSI_{year}.zarr already exists. Skipping.")
            continue
        print("Calculating stress index for year", year)
        ds = xr.open_zarr(f"../data/intermediate/{year}.zarr", decode_coords="all")
        if not hasattr(TAW, "chunks"):
            TAW = TAW.chunk({k: ds.chunks[k] for k in TAW.dims})
        waterstress = (
            (ds.soil_depletion * 100 / TAW).mean("layer").rename("waterstress")
        )
        max_air_temp = xr.open_zarr(
            f"../data/input/{year}.zarr", decode_coords="all"
        ).max_air_temp.astype("f4")
        data_collection = xr.merge(
            [waterstress, max_air_temp, ds.Kc_factor.astype("f4")]
        )
        csi = set_combined_stress_meta(
            data_collection.map_blocks(
                calc_combined_stress, template=data_collection.Kc_factor.astype("f4")
            )
        )
        csi.drop_encoding().to_zarr(f"../data/intermediate/CSI_{year}.zarr", mode="a-")


def set_yield_meta(da: xr.DataArray) -> xr.DataArray:
    """Set metadata for the combined stress index"""
    return da.rename("yield_expectation").assign_attrs(
        dict(
            unit="t/ha",
            long_name="Expected yield in tonnes per hectare",
            description="The expected yield given a certain combined stress "
            "which is crop specific and bases on water availability "
            "and heat above defined thresholds.",
        )
    )


def calc_yield(csi: xr.DataArray) -> xr.DataArray:
    """Estimate yield

    :param csi: Combined stress index
    :type csi: xr.DataArray
    :return: Yield expectation
    :rtype: xr.DataArray
    """
    const = xr.DataArray(
        [6.64, 5.11, 10.99, 87.53],
        coords={"crop": ["winter wheat", "spring barley", "maize", "grassland"]},
    )
    trend = xr.zeros_like(const) - [0.000084, 0.0002, 0.0005, 0.0055]
    yield_expectation = (const + trend * csi.sum("time")).where(
        (~csi.isnull()).any("time")
    )
    return set_yield_meta(yield_expectation)


def main_yield(years: Iterable[int]):
    """Load input data and write yield expectations to Zarr store

    :param years: List years to compute
    :type years: Iterable[int]
    """
    for year in years:
        if os.path.isdir(f"../data/output/{year}.zarr"):
            print(f"! WARNING: {year}.zarr already exists. Skipping.")
            continue
        print("Calculating yield expectation for year", year)
        csi = xr.open_zarr(
            f"../data/intermediate/CSI_{year}.zarr", decode_coords="all"
        ).combined_stress
        yield_expectation = (
            csi.map_blocks(calc_yield, template=csi.isel(time=0).drop_vars("time"))
            .rename("yield_expectation")
            .assign_attrs(
                dict(
                    unit="t/ha",
                    long_name="Expected yield in tonnes per hectare",
                    description="The expected yield given a certain combined stress which is crop "
                    "specific and bases on water availability and heat above defined "
                    "thresholds.",
                )
            )
        )
        yield_expectation.drop_encoding().to_zarr(
            f"../data/output/{year}.zarr", mode="a-"
        )


def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="computes stress and/or yield")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["stress", "yield", "both", "auto"],
        default="auto",
        help="choose whether to compute stress, yield, or both",
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
        default="1Gb",
        help='memory per worker, e.g. "5.67Gb"',
    )
    args = parser.parse_args()
    args.years = sorted(args.years)

    if args.mode == "auto":
        if all(
            os.path.isdir(f"../data/intermediate/CSI_{year}.zarr")
            for year in args.years
        ):
            print(
                "Stress index is present, assuming you want to have the yield expectations "
                "computed."
            )
            args.mode = "yield"
        else:
            print(
                "Stress index is missing for year(s):",
                ", ".join(
                    [
                        str(year)
                        for year in args.years
                        if not os.path.isdir(f"../data/intermediate/CSI_{year}.zarr")
                    ]
                )
                + ".",
                "Computing these first before estimating the yield.",
            )
            args.mode = "both"

    from dask.distributed import LocalCluster, Client

    print("Starting dask")
    client = Client(
        LocalCluster(
            n_workers=args.workers, memory_limit=args.mem_per_worker, death_timeout=30
        )
    )
    print("... access the dashboard at", client.dashboard_link)

    try:
        if args.mode in ["stress", "both"]:
            main_combined_stress(args.years)
        main_yield(args.years)
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


if __name__ == "__main__":
    main_cli()
