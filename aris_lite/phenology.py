#!/usr/bin/env python

"""Phenology module

This module computes the crop coefficients (Kc) and plant heights for different crops
based on temperature and crop-specific phenological rules. These variables are essential
for modeling crop growth, water use, and yield.
"""

__all__ = [
    "main",
    "compute_phenology_variables",
]

from aris_lite import T_crop_names
from dask import array as dask_arr
import numpy as np
import operator
import os
import pandas as pd
from typing import Iterable
import xarray as xr


class Kc_condition_atom:
    """
    Internal comparator class for phenology conditions.

    Represents a single comparison operation (e.g., >= threshold) to be applied
    to a DataArray, supporting both temporal and numeric comparisons.

    :param comparator: Comparison function (e.g., operator.ge).
    :type comparator: callable
    :param value: Value to compare against (float, pd.Timestamp, or xr.DataArray).
    :type value: float | pd.Timestamp | xr.DataArray
    """

    def __init__(
        self, comparator: callable, value: float | pd.Timestamp | xr.DataArray
    ):
        self.comparator = comparator
        self.value = value

    def __str__(self):
        return (
            "Custom condition(comparator: "
            f"{self.comparator}, reference value: {self.value})"
        )

    @property
    def is_temporal(self) -> bool:
        return isinstance(
            self.value, pd.Timestamp
        ) or pd.api.types.is_datetime64_any_dtype(self.value)

    def compare(self, other: xr.DataArray) -> xr.DataArray:
        if self.is_temporal:
            years = np.unique(other.time.dt.year)
            assert years.shape == (
                1,
            )  # implement handling longer time series if relevant
            if isinstance(self.value, pd.Timestamp):
                comp_val = self.value.replace(year=years[0])
            else:
                # self.value is derived from dataset; year does not need to be adapted
                comp_val = self.value
            return self.comparator(other.time.broadcast_like(other), comp_val)
        else:
            return self.comparator(other, self.value)


class Kc_condition:
    """
    Internal class to combine multiple Kc_condition_atom comparators.

    Allows combining several conditions (e.g., after a date AND above a threshold)
    to define phenological stages.

    :param condition_tuple: Tuple of Kc_condition_atom instances.
    :type condition_tuple: tuple[Kc_condition_atom]
    """

    def __init__(self, condition_tuple: tuple["Kc_condition_atom"]):
        self.condition_tuple = condition_tuple

    def __str__(self):
        return "\n".join(
            [
                "Custom condition containing condition atoms:",
                *[str(c) for c in self.condition_tuple],
            ]
        )

    def compare(self, other: xr.DataArray) -> xr.DataArray:
        # True if all conditions met
        return xr.concat(
            [condition_atom.compare(other) for condition_atom in self.condition_tuple],
            dim="temporary_dimension",
        ).all("temporary_dimension")


def conditional_cumulative_temperature(
    temperature: xr.DataArray,
    start_month: int,
    basis_temperature: float,
    start_growing_season: tuple[int, float],
) -> xr.DataArray:
    """
    Calculate cumulative temperature above a threshold.

    Only counts days where the temperature exceeds the basis temperature for a
    minimum number of consecutive days as specified by start_growing_season.
    Used to determine phenological stage transitions.

    :param temperature: Daily temperature time series (one year).
    :type temperature: xr.DataArray
    :param start_month: Month in which to start counting (1-12).
    :type start_month: int
    :param basis_temperature: Temperature threshold for counting.
    :type basis_temperature: float
    :param start_growing_season: Tuple of (minimum consecutive days above threshold,
        threshold value).
    :type start_growing_season: tuple[int, float]
    :return: Cumulative temperature above threshold.
    :rtype: xr.DataArray
    """
    return xr.where(
        np.logical_and(
            #  this intransparent lines (until end) satisfy the requirement to
            #  have a number of consecutive days with temperatures above a
            #  threshold
            np.logical_and(
                temperature.time.dt.month >= start_month,
                (temperature >= start_growing_season[1])
                .isel(time=slice(None, None, -1))
                .rolling(time=start_growing_season[0])
                .sum()
                == start_growing_season[0],
            ).cumsum("time")
            >= 1,  # end
            temperature >= basis_temperature,
        ),
        temperature - basis_temperature,
        0,
    ).cumsum("time")


def apply_condition_value_list(
    condition_value_list: Iterable[tuple["Kc_condition", float]],
    arr: xr.DataArray,
) -> xr.DataArray:
    """
    Assign values to an array based on a list of (condition, value) pairs.

    Later values override earlier ones. Used to set Kc or plant height values
    for different phenological stages.

    :param condition_value_list: List of (condition, value) pairs.
    :type condition_value_list: Iterable[tuple[Kc_condition, float]]
    :param arr: Input DataArray to assign values to.
    :type arr: xr.DataArray
    :return: DataArray with assigned values where conditions are met.
    :rtype: xr.DataArray
    """
    out = xr.DataArray(np.nan, coords=arr.coords)
    for cond, val in condition_value_list:
        out = xr.where(cond.compare(arr), val, out)
    return out


def build_Kc_factor_array(
    Kc_factor_defs: Iterable[tuple["Kc_condition", float]],
    arr: xr.DataArray,
) -> xr.DataArray:
    """
    Interpolate Kc factor values based on phenological stage definitions.

    Applies condition-value pairs and linearly interpolates missing values over time.

    :param Kc_factor_defs: Iterable of (condition, value) pairs for Kc.
    :type Kc_factor_defs: Iterable[tuple[Kc_condition, float]]
    :param arr: Values DataArray that is compared against conditions.
    :type arr: xr.DataArray
    :return: Interpolated Kc factor DataArray.
    :rtype: xr.DataArray
    """
    return apply_condition_value_list(Kc_factor_defs, arr).interpolate_na(
        "time", "linear"
    )


def build_plant_height_array(
    plant_height_defs: Iterable[tuple["Kc_condition", float]],
    arr: xr.DataArray,
) -> xr.DataArray:
    """
    Assign plant height values based on phenological stage definitions.

    Applies condition-value pairs and fills missing values with zero.

    :param plant_height_defs: Iterable of (condition, value) pairs for plant height.
    :type plant_height_defs: Iterable[tuple[Kc_condition, float]]
    :param arr: Values DataArray that is compared against conditions.
    :type arr: xr.DataArray
    :return: Plant height DataArray.
    :rtype: xr.DataArray
    """
    return (
        apply_condition_value_list(plant_height_defs, arr)
        .interpolate_na("time", "linear")
        .fillna(0)
    )


def compute_phenology_variables(
    temperature: xr.DataArray,
    crops: Iterable[T_crop_names] = (
        "winter wheat",
        "spring barley",
        "maize",
        "grassland",
    ),
) -> xr.Dataset:
    """
    Compute crop coefficients (Kc) and plant heights for specified crops.

    Uses temperature data and crop-specific rules to determine phenological stages,
    then assigns Kc and plant height values accordingly.

    :param temperature: Daily surface air temperature average for one year.
    :type temperature: xr.DataArray
    :param crops: List of crops to compute phenology for.
    :type crops: Iterable[str], optional
    :return: Dataset containing Kc_factor and plant_height for each crop.
    :rtype: xr.Dataset

    Notes
    -----
    The current implementation draws on the three references below
    (basic idea: [1], parameters and specifications: [2], grass cut
    days: [3]).
    [1] Allen, R. G. (Ed.). (2000). Crop evapotranspiration: Guidelines
        for computing crop water requirements (repr). Food and
        Agriculture Organization of the United Nations.
    [2] Eitzinger, J., Daneu, V., Kubu, G., Thaler, S., Trnka, M.,
        Schaumberger, A., Schneider, S., & Tran, T. M. A. (2024). Grid
        based monitoring and forecasting system of cropping conditions
        and risks by agrometeorological indicators in Austria –
        Agricultural Risk Information System ARIS. Climate Services, 34,
        100478. https://doi.org/10.1016/j.cliser.2024.100478
    [3] Schaumberger, A. (2011). Räumliche Modelle zur Vegetations- und
        Ertragsdynamik im Wirtschaftsgrünland [Dissertation, Graz
        University of Technology].
        https://repository.tugraz.at/publications/npc97-y3058.
    """
    # all of winter wheat, spring barley, grain maize, potato, soybeans and
    # grassland (mähwiese) need to be included

    # TODO search xclim for degree_day_exceedance !
    # TODO CRS should be adopted from coords

    Kc_ini_val = 0.4
    Kc_mid_val = 1.2
    Kc_end_val = 0.5
    Kc_out_val = 0.4

    before_growing_season = Kc_condition_atom(operator.eq, 0)
    before_out_season = Kc_condition_atom(
        operator.lt, pd.Timestamp(month=12, day=1, year=999)
    )
    out_season = Kc_condition_atom(operator.ge, pd.Timestamp(month=12, day=1, year=999))

    Kc_factor_da_list = []
    plant_height_da_list = []
    for crop in crops:
        if crop == "winter wheat":
            mid_season_start_cumT = 350
            mid_season_end_cumT = mid_season_start_cumT + 692
            cumT = conditional_cumulative_temperature(
                temperature,
                start_month=3,
                basis_temperature=5,
                start_growing_season=(5, 5),
            )
        elif crop == "spring barley":
            mid_season_start_cumT = 502
            mid_season_end_cumT = mid_season_start_cumT + 568
            cumT = conditional_cumulative_temperature(
                temperature,
                start_month=3,
                basis_temperature=5,
                start_growing_season=(5, 5),
            )
        elif crop == "maize":
            mid_season_start_cumT = 249
            mid_season_end_cumT = mid_season_start_cumT + 1238
            cumT = conditional_cumulative_temperature(
                temperature,
                start_month=4,
                basis_temperature=8,
                start_growing_season=(5, 10),
            )
        elif crop.startswith("wofost potato"):
            Kc_mid_val = 1.0
            cumT = (
                temperature.where(temperature >= 3, 0)
                .where(
                    temperature.time
                    >= pd.Timestamp(
                        year=temperature.time.dt.year.item(0), month=4, day=15
                    )
                )
                .cumsum("time")
            )
            before_growing_season_cumT = 170
            before_growing_season = Kc_condition_atom(
                operator.lt, before_growing_season_cumT
            )
            if crop.endswith("very early"):
                mid_season_start_cumT = before_growing_season_cumT + 150
                mid_season_end_cumT = mid_season_start_cumT + 1250
            elif crop.endswith("mid"):
                mid_season_start_cumT = before_growing_season_cumT + 150
                mid_season_end_cumT = mid_season_start_cumT + 1500
            elif crop.endswith("late"):
                mid_season_start_cumT = before_growing_season_cumT + 200
                mid_season_end_cumT = mid_season_start_cumT + 1700
            else:
                print(f"! WARNING: requested crop {crop} not recognized; skipped")
                continue
        elif crop == "grassland":
            """
            To be consistent with the original ARIS, grassland is
            implemented following the Dissertation by Schaumberger [1].
            First: the temperature sums are evaluated at given dates
            Second: the cutting date is determined based on a predefined
            temperature sum to time difference mapping
            The algorithm is complex compared to the earlier crops. For
            details refer to [1].
            [1] Schaumberger, A. (2011). Räumliche Modelle zur
                Vegetations- und Ertragsdynamik im Wirtschaftsgrünland
                [Dissertation, Graz University of Technology].
                https://repository.tugraz.at/publications/npc97-y3058.

            """
            earliest = [
                [141, 250],
                [125, 161, 229],
                [122, 154, 193, 239],
            ]
            regular = [
                [168, 274],
                [148, 206, 271],
                [139, 181, 227, 277],
            ]
            latest = [
                [200, 307],
                [177, 260, 301],
                [155, 225, 269, 305],
            ]
            cumTs = [
                [630, 710],
                [630, 710, 910],
                [630, 710, 910, 850],
            ]
            cumT = conditional_cumulative_temperature(
                temperature,
                start_month=3,
                basis_temperature=0,
                start_growing_season=(5, 5),
            )
            # Kc and plant height values
            Kc_ini_val = 0.4
            Kc_mid_val = 1.2
            Kc_end_val = 0.9
            Kc_out_val = 0.4
            # plH_ini_val = 0
            plH_mid_val = 0.7
            plH_end_val = 0.2
            plH_out_val = 0.2  # debatable choice
            before_out_season = Kc_condition_atom(
                operator.lt, pd.Timestamp(month=11, day=1, year=999)
            )
            out_season = Kc_condition_atom(
                operator.ge, pd.Timestamp(month=11, day=1, year=999)
            )
            group_output_collector = []
            group_output_collector2 = []
            try:
                for (
                    _,
                    group,
                ) in cumT.groupby_bins(  # FIXME wrap in .map_blocks if chunked
                    cumT.sel(time=f"{cumT.time[0].dt.year.values}-10-31"),
                    [sum(sublist) for sublist in cumTs] + [99999],
                ):
                    before_growing_season = Kc_condition_atom(
                        operator.lt,
                        cumT[(cumT == 0).argmin("time").compute()].time.dt.dayofyear,
                    )
                    before_march_first = Kc_condition_atom(
                        operator.lt,
                        cumT[:, 0]
                        .sel(time=f"{cumT.time.dt.year.item(0)}-03-01")
                        .time.dt.dayofyear.item(0),
                    )
                    # init list to define stages dynamically
                    Kc_factor_periods = [
                        (before_growing_season, Kc_ini_val),
                        (before_march_first, Kc_out_val),
                    ]
                    plant_height_periods = [
                        (before_growing_season, plH_out_val),
                        (before_march_first, plH_out_val),
                    ]
                    # cycle through threshold values
                    for T_threshold, mid, lower, upper in zip(
                        cumTs.pop(0), regular.pop(0), earliest.pop(0), latest.pop(0)
                    ):
                        lower_fraction_limit = 0.5 if mid < 170 else 0.4
                        T_sum_ratio = (
                            group.where(cumT.time.dt.dayofyear == mid).sum("time")
                            / T_threshold
                        ).clip(lower_fraction_limit, 2)
                        cut_doy = xr.where(
                            T_sum_ratio < 1,  # if smaller 1 then delayed, else earlier
                            np.round(
                                upper
                                - (upper - mid)
                                * (T_sum_ratio - lower_fraction_limit)
                                / (1 - lower_fraction_limit)
                            ).astype("int"),
                            np.round(mid - (mid - lower) * (T_sum_ratio - 1)).astype(
                                "int"
                            ),
                        )
                        cond_cut_doy = Kc_condition_atom(operator.eq, cut_doy - 1)
                        cond_just_after_cut = Kc_condition_atom(operator.eq, cut_doy)
                        Kc_factor_periods.extend(
                            [
                                (cond_cut_doy, Kc_mid_val),
                                (cond_just_after_cut, Kc_ini_val),
                            ]
                        )
                        plant_height_periods.extend(
                            [
                                (cond_cut_doy, plH_mid_val),
                                (cond_just_after_cut, plH_end_val),
                            ]
                        )
                        group = group - group.where(
                            cumT.time.dt.dayofyear == cut_doy - 1
                        ).sum("time")
                    # set end state
                    # `ge` to match the original ARIS should be `gt`
                    cond_after_cut = Kc_condition_atom(operator.ge, cut_doy)
                    Kc_factor_periods.extend(
                        [
                            (cond_after_cut, Kc_end_val),
                            (out_season, Kc_out_val),
                        ]
                    )
                    group_output_collector.append(
                        build_Kc_factor_array(
                            Kc_factor_periods,
                            group.time.dt.dayofyear.broadcast_like(group),
                        )
                    )
                    plant_height_periods.extend(
                        [
                            (cond_after_cut, plH_end_val),
                            (out_season, plH_out_val),
                        ]
                    )
                    group_output_collector2.append(
                        build_plant_height_array(
                            plant_height_periods,
                            group.time.dt.dayofyear.broadcast_like(group),
                        )
                    )
                Kc_factor_da_list.append(
                    xr.concat(group_output_collector, group_output_collector[0].dims[1])
                    .sortby(group_output_collector[0].dims[1])
                    .unstack()
                    .reindex_like(cumT)
                    .assign_coords({coord: vals for coord, vals in cumT.coords.items()})
                    .rename(crop.replace(" ", "_"))
                )
                plant_height_da_list.append(
                    xr.concat(
                        group_output_collector2, group_output_collector[0].dims[1]
                    )
                    .sortby(group_output_collector[0].dims[1])
                    .unstack()
                    .reindex_like(cumT)
                    .assign_coords({coord: vals for coord, vals in cumT.coords.items()})
                    .rename(crop.replace(" ", "_"))
                )
            except ValueError as err:
                if str(err).startswith("None of the data falls within bins with edges"):
                    for da_list in [Kc_factor_da_list, plant_height_da_list]:
                        if da_list[-1].name != "grassland":
                            da_list.append(
                                xr.DataArray(np.nan, coords=cumT.coords).rename(
                                    crop.replace(" ", "_")
                                )
                            )
                else:
                    raise err
            continue
        else:
            print(
                f"! WARNING: requested crop {crop} was not recognized and is skipped."
            )
            continue

        after_mid_season_start = Kc_condition_atom(operator.ge, mid_season_start_cumT)
        before_mid_season_end = Kc_condition_atom(operator.le, mid_season_end_cumT)
        EGS_date = cumT[
            before_mid_season_end.compare(cumT).argmin("time").compute()
        ].time
        # set EGS_date to nan where applicable
        EGS_date = EGS_date.where(
            EGS_date > pd.Timestamp(month=3, day=1, year=cumT.time[0].dt.year.values)
        )
        before_EGS = Kc_condition_atom(operator.lt, EGS_date)
        # after_EGS = Kc_condition_atom(operator.ge, EGS_date + pd.Timedelta(days=1))
        mid_season = Kc_condition([after_mid_season_start, before_EGS])
        # late_and_end_season = Kc_condition([after_EGS, before_out_season])
        before_late_end = Kc_condition_atom(
            operator.lt, EGS_date + pd.Timedelta(days=14)
        )
        mid_and_late = Kc_condition([after_mid_season_start, before_late_end])
        after_late_season = Kc_condition_atom(
            operator.ge, EGS_date + pd.Timedelta(days=14)
        )
        end_season = Kc_condition([after_late_season, before_out_season])
        # late_and_end = Kc_condition([after_EGS, before_out_season])

        Kc_factor_periods = [
            (before_growing_season, Kc_ini_val),
            (mid_season, Kc_mid_val),
            (end_season, Kc_end_val),
            (out_season, Kc_out_val),
        ]
        Kc_factor_da_list.append(
            build_Kc_factor_array(Kc_factor_periods, cumT).rename(
                crop.replace(" ", "_")
            )
        )

        if crop in ["winter wheat", "spring barley"]:
            plant_height_periods = [
                (before_growing_season, 0),
                (mid_and_late, 1),
                # (end_season, 0.2),
            ]
        elif crop == "maize":
            plant_height_periods = [
                (before_growing_season, 0),
                (mid_and_late, 2),
                # (end_season, 0.2),
            ]
        elif "potato" in crop:
            plant_height_periods = [
                (before_growing_season, 0),
                (mid_and_late, 0.6),
                # (end_season, 0),
            ]
        elif crop == "grassland":
            plant_height_periods = [(end_season, 0.2)]  # inconsistent with the above
        else:
            raise Exception(
                "If you see this error, implement plant height for missing crop."
            )
        plant_height_da_list.append(
            build_plant_height_array(plant_height_periods, cumT).rename(
                crop.replace(" ", "_")
            )
        )

    Kc_factor_da_list = (
        xr.concat(Kc_factor_da_list, "crop")
        .assign_coords(crop=("crop", list(crops)))
        .rename("Kc_factor")
    )
    plant_height_da_list = (
        xr.concat(plant_height_da_list, "crop")
        .assign_coords(crop=("crop", list(crops)))
        .rename("plant_height")
    )
    out = xr.merge([Kc_factor_da_list, plant_height_da_list])
    return out


def main(
    years: Iterable[int],
    crops: Iterable[T_crop_names] = (
        "winter wheat",
        "spring barley",
        "maize",
        "grassland",
    ),
):
    """
    Load data, compute phenology variables, and save output for specified years.

    For each year, loads temperature data, computes phenology variables for the given
    crops, and writes the results to a Zarr store.

    :param years: List of years to compute.
    :type years: Iterable[int]
    :param crops: List of crops to compute, defaults to ("winter wheat",
                  "spring barley", "maize", "grassland").
    :type crops: Iterable, optional
    """
    for year in years:
        if os.path.isdir(f"../data/intermediate/{year}.zarr"):
            print(f"! WARNING: {year}.zarr already exists. Skipping.")
            continue
        print("Calculating phenology variables for year", year, "and crops", crops)
        T2m = xr.open_zarr(
            f"../data/input/{year}.zarr", decode_coords="all"
        ).air_temperature
        if T2m.time.dt.calendar in [
            "noleap",
        ]:
            original_calendar = T2m.time.dt.calendar
            T2m = xr.coding.calendar_ops.convert_calendar(T2m, "gregorian")
        template = xr.DataArray(
            dask_arr.zeros(shape=(len(crops), *T2m.shape), dtype="f4"),
            coords=T2m.expand_dims({"crop": crops}).coords,
        ).chunk(dict(crop=-1, time=-1, x=41, y=37))
        template = xr.merge(
            [template.rename("Kc_factor"), template.rename("plant_height")]
        )
        result = T2m.map_blocks(
            lambda x: compute_phenology_variables(x, crops), template=template
        )
        if "original_calendar" in locals():
            result = xr.coding.calendar_ops.convert_calendar(result, original_calendar)
        result.drop_encoding().to_zarr(f"../data/intermediate/{year}.zarr", mode="a-")


def main_cli():
    """
    Command-line interface for computing phenology variables.

    Parses command-line arguments to determine which years to process and how
    many Dask workers to use. Initializes a Dask cluster for parallel
    processing, handles missing data, and manages workflow for phenology
    calculations.

    Usage:
        aris-calc-pheno [years ...] [--workers N] [--mem-per-worker SIZE]

    :return: None
    """
    import argparse

    parser = argparse.ArgumentParser(description="computes stress and/or yield")
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
        default="3Gb",
        help='memory per worker, e.g. "5.67Gb"',
    )
    args = parser.parse_args()
    args.years = sorted(args.years)

    from dask.distributed import LocalCluster, Client

    print("Starting dask")
    client = Client(
        LocalCluster(
            n_workers=args.workers, memory_limit=args.mem_per_worker, death_timeout=30
        )
    )
    print("... access the dashboard at", client.dashboard_link)

    try:
        main(args.years)
    except (FileNotFoundError,) as err:
        if str(err).startswith("Unable to find group"):
            print(
                "\n! ERROR: data missing. Verify that the necessary data are "
                "available.\n"
            )
            raise
    finally:
        client.close()
        print("Closed dask client\n")

    print("Sucessfully computed phenology related variables!\n")
    print(
        "Continue by computing the soil water by running\n\t`python water_budget.py"
        "-m soil [year1 ...]`\n"
    )


if __name__ == "__main__":
    main_cli()
