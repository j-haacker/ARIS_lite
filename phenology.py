# TODO add module discription

import operator
import numpy as np
import pandas as pd
import xarray as xr


__all__ = [
    "calculate_Kc_factors",
    "conditional_cumulative_temperature",
    "build_Kc_factor_array",
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


def conditional_cummulative_temperature(temperature: xr.DataArray,
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


def build_Kc_factor_array(Kc_factor_defs: list[tuple["Kc_condition", float]],
                          cumT: xr.DataArray,
                          ) -> xr.DataArray:
    # TODO add docstring (note: Think: FIFO stack. As a consequence, later values override previous.)
    out = xr.DataArray(np.nan, coords=cumT.coords)
    for cond, val in Kc_factor_defs:
        out = xr.where(cond.compare(cumT), val, out)
    return out.chunk(dict(time=-1)).interpolate_na("time", "linear")


def calculate_Kc_factors(temperature,
                         crop_list: list[str] = None,
                         ) -> xr.Dataset:
    # TODO add docstring

    # TODO CRS should be adopted from coords

    if crop_list is None:
        crop_list = ["winter wheat", "spring barley", "maize"]

    before_growing_season = Kc_condition_atom(operator.eq, 0)
    before_out_season = Kc_condition_atom(operator.lt, pd.Timestamp(month=12, day=1, year=999))
    out_season = Kc_condition_atom(operator.ge, pd.Timestamp(month=12, day=1, year=999))

    cumT_5 = conditional_cummulative_temperature(temperature, start_month=3, threshold=5,
                                                 timesteps_above_threshold=5)
    cumT_8 = conditional_cummulative_temperature(temperature, start_month=4, threshold=8,
                                                 timesteps_above_threshold=5)

    out = []
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
            # because the definde thresholds result in no grassland at all, I use imaginary values
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
                out.append(xr.concat(group_output_collector, "stacked_y_x").sortby("stacked_y_x")
                             .unstack().reindex_like(cumT_5).rename(crop.replace(" ", "_")))
            except ValueError as err:
                if str(err).startswith("None of the data falls within bins with edges"):
                    out.append(xr.DataArray(np.nan, coords=cumT_5.coords)
                                 .rename(crop.replace(" ", "_")))
                else:
                    raise err
            finally:
                continue
        else:
            print(f"! WARNING: requested crop {crop} was not recognized and is skipped.")

        after_mid_season_start = Kc_condition_atom(operator.ge, mid_season_start_cumT)
        before_mid_season_end = Kc_condition_atom(operator.le, mid_season_end_cumT)
        EGS_date = cumT[before_mid_season_end.compare(cumT).argmin("time").compute()].time
        # set EGS_date to nan where applicable
        EGS_date = EGS_date.where(EGS_date > pd.Timestamp(month=3, day=1,
                                                          year=cumT.time[0].dt.year.values))
        before_EGS = Kc_condition_atom(operator.lt, EGS_date+pd.Timedelta(days=1))
        mid_season = Kc_condition([after_mid_season_start, before_EGS])
        after_late_start = Kc_condition_atom(operator.ge, EGS_date+pd.Timedelta(days=15))
        end_season = Kc_condition([after_late_start, before_out_season])

        Kc_factor_periods = [
            (before_growing_season, .4),
            (mid_season, 1.2),
            (end_season, .5),
            (out_season, .4)
        ]
        out.append(
            build_Kc_factor_array(Kc_factor_periods, cumT).rename(crop.replace(" ", "_"))
        )

    return xr.concat(out, "crop").assign_coords(crop=crop_list).rename("Kc_factor")
