# TODO add module discription

from dask import array as dask_arr
import numpy as np
import operator
import os
import pandas as pd
import sys
import xarray as xr


__all__ = [
    "compute_phenology_variables",
    "conditional_cumulative_temperature",
    "build_Kc_factor_array",
    "build_plant_height_array",
    "Kc_condition",
    "Kc_condition_atom",
]


class Kc_condition_atom:
    # TODO add docstring
    def __init__(self, comparator: callable, value: float | pd.Timestamp | xr.DataArray):
        self.comparator = comparator
        self.value = value

    @property
    def is_temporal(self):
        return (isinstance(self.value, pd.Timestamp)
                or pd.api.types.is_datetime64_any_dtype(self.value))

    def compare(self, other: xr.DataArray):
        if self.is_temporal:
            years = np.unique(other.time.dt.year)
            assert years.shape == (1,)  # implement handling longer time series if relevant
            if isinstance(self.value, pd.Timestamp):
                comp_val = self.value.replace(year=years[0])
            else:
                # self.value is derived from dataset and year does not need to be adapted
                comp_val = self.value
            return self.comparator(other.time.broadcast_like(other), comp_val)
        else:
            return self.comparator(other, self.value)


class Kc_condition:
    # TODO add docstring
    def __init__(self, condition_tuple: tuple["Kc_condition_atom"]):
        self.condition_tuple = condition_tuple

    def compare(self, other: xr.DataArray):
        # True if all conditions met
        return xr.concat(
            [condition_atom.compare(other) for condition_atom in self.condition_tuple],
            dim="temporary_dimension"
        ).all("temporary_dimension")


def conditional_cumulative_temperature(temperature: xr.DataArray,
                                       start_month: int,
                                       threshold: float,
                                       timesteps_above_threshold: int = 5,
                                       ) -> xr.DataArray:
    # TODO add docstring
    return xr.where(
        np.logical_and(
            np.logical_and(
                temperature.time.dt.month >= start_month,
                (temperature >= threshold).isel(time=slice(None, None, -1))
                                          .rolling(time=timesteps_above_threshold).sum()
                == timesteps_above_threshold
            ).cumsum("time") >= 1,
            temperature >= threshold
        ), temperature-threshold, 0).cumsum("time")


def apply_condition_value_list(condition_value_list: list[tuple["Kc_condition", float]],
                               arr: xr.DataArray,
                               ) -> xr.DataArray:
    # TODO add docstring (note: Think: FIFO stack. As a consequence, later values override previous.)
    out = xr.DataArray(np.nan, coords=arr.coords)
    for cond, val in condition_value_list:
        out = xr.where(cond.compare(arr), val, out)
    return out


def build_Kc_factor_array(Kc_factor_defs: list[tuple["Kc_condition", float]],
                          cumT: xr.DataArray,
                          ) -> xr.DataArray:
    return apply_condition_value_list(Kc_factor_defs, cumT).interpolate_na("time", "linear")


def build_plant_height_array(plant_height_defs: list[tuple["Kc_condition", float]],
                             cumT: xr.DataArray,
                             ) -> xr.DataArray:
    return apply_condition_value_list(plant_height_defs, cumT).fillna(0)


def compute_phenology_variables(temperature: xr.DataArray,
                                crop_list: list[str] = None,
                                ) -> xr.Dataset:
    # TODO add docstring

    # all of winter wheat, spring barley, grain maize, potato, soybeans and a grassland (mähwiese)
    # need to be included

    # TODO CRS should be adopted from coords

    if crop_list is None:
        crop_list = ["winter wheat", "spring barley", "maize", "grassland"]

    before_growing_season = Kc_condition_atom(operator.eq, 0)
    before_out_season = Kc_condition_atom(operator.lt, pd.Timestamp(month=12, day=1, year=999))
    out_season = Kc_condition_atom(operator.ge, pd.Timestamp(month=12, day=1, year=999))

    cumT_5 = conditional_cumulative_temperature(temperature, start_month=3, threshold=5,
                                                timesteps_above_threshold=5)
    cumT_8 = conditional_cumulative_temperature(temperature, start_month=4, threshold=8,
                                                timesteps_above_threshold=5)

    Kc_factor_da_list = []
    plant_height_da_list = []
    for crop in crop_list:
        if crop == "winter wheat":
            mid_season_start_cumT = 350
            mid_season_end_cumT = mid_season_start_cumT+692
            cumT = cumT_5
        elif crop == "spring barley":
            mid_season_start_cumT = 502
            mid_season_end_cumT = mid_season_start_cumT+568
            cumT = cumT_5
        elif crop == "maize":
            mid_season_start_cumT = 249
            mid_season_end_cumT = mid_season_start_cumT+1238
            cumT = cumT_8
        elif crop == "grassland":
            # grassland needs to be implemented slightly different
            # ! cumulative temperature thresholds do not seem to make sense because
            #   1) 2-cuts require a warmer year than 3-cuts but are only applied in colder years
            #   2) all cut strategies only differ slightly in their end temperature sums
            Kc_out_val = 0.2
            Kc_ini_val = 0.4
            # because the defined thresholds result in no grassland at all, I use imaginary values
            # for testing
            # cumTs = [np.cumsum(thresholds) for thresholds in [
            #     [1170, 1800],
            #     [770, 1020, 1260],
            #     [630, 710, 910, 850]
            # ]]
            cumTs = [np.cumsum(thresholds) for thresholds in [
                [870, 800],
                [770, 820, 860],
                [630, 710, 710, 750]
            ]]
            group_output_collector = []
            group_output_collector2 = []
            try:
                for label, group in cumT_5.groupby_bins(  # FIXME wrap in .map_blocks if chunked
                    cumT_5.sel(time=f"{cumT_5.time[0].dt.year.values}-11-30"),
                    [sublist[-1] for sublist in cumTs]+[99999]
                ):
                    Kc_factor_periods = [
                        (before_growing_season, Kc_ini_val),
                        (Kc_condition_atom(operator.lt, pd.Timestamp(month=3, day=1, year=999)),
                         Kc_out_val),
                    ]
                    for cumT_threshold in cumTs.pop(0):
                        tmp_EGS = group[(group < cumT_threshold).argmin("time").compute()].time
                        Kc_factor_periods.extend([
                            (Kc_condition_atom(operator.eq, tmp_EGS), 1.2),
                            (Kc_condition_atom(operator.eq, tmp_EGS+pd.Timedelta(days=1)), 0.4),
                        ])
                    Kc_factor_periods.extend([
                        (Kc_condition_atom(operator.ge, tmp_EGS+pd.Timedelta(days=1)), 0.4),
                        (out_season, Kc_out_val)
                    ])
                    group_output_collector.append(build_Kc_factor_array(Kc_factor_periods, group))
                    end_season = Kc_condition([Kc_condition_atom(operator.ge, tmp_EGS+pd.Timedelta(days=1)),
                                               before_out_season])
                    group_output_collector2.append(build_plant_height_array([(end_season, 0.2)], group))
                Kc_factor_da_list.append(xr.concat(group_output_collector, "stacked_y_x").sortby("stacked_y_x")
                                           .unstack().reindex_like(cumT_5).rename(crop.replace(" ", "_")))
                plant_height_da_list.append(xr.concat(group_output_collector2, "stacked_y_x").sortby("stacked_y_x")
                                              .unstack().reindex_like(cumT_5).rename(crop.replace(" ", "_")))
            except ValueError as err:
                if str(err).startswith("None of the data falls within bins with edges"):
                    Kc_factor_da_list.append(xr.DataArray(np.nan, coords=cumT_5.coords)
                                               .rename(crop.replace(" ", "_")))
                    plant_height_da_list.append(xr.DataArray(np.nan, coords=cumT_5.coords)
                                                  .rename(crop.replace(" ", "_")))
                else:
                    raise err
            finally:
                continue
        else:
            print(f"! WARNING: requested crop {crop} was not recognized and is skipped.")
            continue

        after_mid_season_start = Kc_condition_atom(operator.ge, mid_season_start_cumT)
        before_mid_season_end = Kc_condition_atom(operator.le, mid_season_end_cumT)
        EGS_date = cumT[before_mid_season_end.compare(cumT).argmin("time").compute()].time
        # set EGS_date to nan where applicable
        EGS_date = EGS_date.where(EGS_date > pd.Timestamp(month=3, day=1,
                                                          year=cumT.time[0].dt.year.values))
        before_EGS = Kc_condition_atom(operator.lt, EGS_date+pd.Timedelta(days=1))
        after_EGS = Kc_condition_atom(operator.ge, EGS_date+pd.Timedelta(days=1))
        mid_season = Kc_condition([after_mid_season_start, before_EGS])
        late_and_end_season = Kc_condition([after_EGS, before_out_season])
        after_late_end = Kc_condition_atom(operator.ge, EGS_date+pd.Timedelta(days=15))
        end_season = Kc_condition([after_late_end, before_out_season])

        Kc_factor_periods = [
            (before_growing_season, .4),
            (mid_season, 1.2),
            (end_season, .5),
            (out_season, .4)
        ]
        Kc_factor_da_list.append(
            build_Kc_factor_array(Kc_factor_periods, cumT).rename(crop.replace(" ", "_"))
        )

        if crop in ["winter wheat", "spring barley"]:
            plant_height_periods = [(mid_season, 1),
                                    (late_and_end_season, .2)]
        elif crop == "maize":
            plant_height_periods = [(mid_season, 2),
                                    (late_and_end_season, .2)]
        elif crop == "grassland":
            plant_height_periods = [(end_season, .2)]
        else:
            raise Exception("If you see this error, implement plant height for missing crop.")
        plant_height_da_list.append(
            build_plant_height_array(plant_height_periods, cumT).rename(crop.replace(" ", "_"))
        )

    Kc_factor_da_list = xr.concat(Kc_factor_da_list, "crop").assign_coords(crop=crop_list).rename("Kc_factor")
    plant_height_da_list = xr.concat(plant_height_da_list, "crop").assign_coords(crop=crop_list).rename("plant_height")
    out = xr.merge([Kc_factor_da_list, plant_height_da_list])
    # print(out, flush=True)
    # raise
    return out


def main(year: int,
         crop_list: list = None) -> None:
    if crop_list is None:
        crop_list = ["winter wheat", "spring barley", "maize", "grassland"]
    print("Calculating phenology variables for year", year, "and crops", crop_list)
    T2m = xr.open_zarr(f"../data/input/{year}.zarr", decode_coords="all").air_temperature
    template = xr.DataArray(dask_arr.zeros(shape=(len(crop_list), *T2m.shape), dtype="f4"),
                            coords=T2m.expand_dims({"crop": crop_list}).coords)\
                 .chunk(dict(crop=-1, time=-1, x=41, y=37))
    template = xr.merge([template.rename("Kc_factor"),
                         template.rename("plant_height")
                         ])
    T2m.map_blocks(lambda x: compute_phenology_variables(x, crop_list), template=template)\
       .drop_encoding().to_zarr(f"../data/intermediate/{year}.zarr", mode="a-")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        year_list = []
        for arg in sys.argv[1:]:
            try:
                year_list.append(int(arg))
                assert year_list[-1] > 1900
                assert year_list[-1] < 2300
            except (ValueError, AssertionError):
                print(f"! WARNING: {arg} is not a valid year. Skipping.")
                continue
        print("Requested years:", year_list)
    else:
        year_list = [2020, 2021, 2023]
        print("Processing default year list:", year_list)
    from dask.distributed import LocalCluster, Client
    client = Client(LocalCluster(4, memory_limit="6.7Gb"))
    print(client.dashboard_link)
    try:
        for year in year_list:
            if os.path.isdir(f"../data/intermediate/{year}.zarr"):
                print(f"! WARNING: {year}.zarr already exists. Skipping.")
                continue
            main(year)
    finally:
        client.close()
        print("Client closed.")
