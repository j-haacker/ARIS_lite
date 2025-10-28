"""Implementation of the ARIS(_lite) phenology model

This package provides functions for simulating crop phenology, water budget, and yield
expectation using environmental and crop-specific data.
"""

__version__ = "0.2.0.dev0"

__all__ = [
    "aris_1go",
    "extract_point_data",
    "phenology",
    "water_budget",
    "yield_expectation",
]

from typing import Literal
import xarray as xr

T_crop_names = Literal[
    "winter wheat",
    "spring barley",
    "maize",
    "grassland",
    "wofost potato very early",
    "wofost potato mid",
    "wofost potato late",
]


def aris_1go(
    ds: xr.Dataset,
    crops: list[T_crop_names],
):
    """
    Run the full ARIS-lite workflow on a single dataset.

    This function sequentially applies snow calculation, phenology computation,
    soil water calculation, water stress, combined stress, and yield estimation
    to the input dataset. It is intended for small datasets and returns a merged
    xarray.Dataset with all computed variables.

    :param ds: Input dataset containing meteorological and crop data.
    :type ds: xr.Dataset
    :return: Dataset with all ARIS-lite output variables.
    :rtype: xr.Dataset
    """
    from aris_lite.water_budget import calc_snow, calc_soil_water
    from aris_lite.phenology import compute_phenology_variables
    from aris_lite.yield_expectation import calc_combined_stress, calc_yield

    def _load_resample_apply(ds, func, *args, **kwargs):
        return (
            ds.load()
            .resample(time="YE")
            .map(func, args=args, **kwargs)
            .assign_coords(time=("time", ds.time.data))
        )

    ds = xr.merge(
        [
            ds,
            calc_snow(
                ds.assign(
                    initial_snowcover=xr.zeros_like(ds.precipitation.isel(time=0))
                )
            ).persist(),
        ]
    )
    ds = xr.merge(
        [
            ds,
            (
                xr.coding.calendar_ops.convert_calendar(
                    ds.air_temperature.where(~(ds.snowcover > 0)), "gregorian"
                ).pipe(
                    _load_resample_apply,
                    compute_phenology_variables,
                    crops,
                )
            ).persist(),
        ]
    )
    ds = xr.merge(
        [
            ds,
            ds.map_blocks(
                _load_resample_apply,
                args=(calc_soil_water,),
                template=xr.Dataset(
                    {
                        var: xr.zeros_like(ds.Kc_factor.broadcast_like(ds.TAW))
                        .transpose(*ds.Kc_factor.dims, ...)
                        .chunk(ds.chunks)
                        for var in ["evapotranspiration", "evapo_ETC", "soil_depletion"]
                    }
                ),
            )
            .chunk(ds.chunks)
            .persist(),
        ]
    )
    ds = ds.assign(
        waterstress=(ds.soil_depletion * 100 / ds.TAW).mean("layer").persist()
    )
    ds = xr.merge([ds, calc_combined_stress(ds).persist()])
    ds = xr.merge(
        [
            ds,
            ds.combined_stress.pipe(_load_resample_apply, calc_yield).persist(),
        ]
    )
    return ds.map(lambda da: da.astype("float32") if da.dtype.kind == "f" else da)


def cli():
    """
    Command-line interface for running the full ARIS-lite workflow.

    Parses command-line arguments for input/output paths and Dask cluster configuration,
    then runs the full workflow and writes the output dataset.

    Usage:
        aris-1go [--workers N] [--mem-per-worker SIZE] input.zarr output.zarr

    :return: None
    """
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent

    parser = ArgumentParser(
        prog="aris-1go",  # TODO add name
        formatter_class=RawDescriptionHelpFormatter,
        description=dedent(
            """Calc all standard ARIS output in a single run
            
            This routine is meant for small datasets. Note that intermediately large
            amounts of data are generated that can quickly exhaust your memory.
            """
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=6, help="number of dask workers"
    )  # TODO change default to 1
    parser.add_argument(
        "--mem-per-worker",
        type=str,
        default="3Gb",
        help='memory per worker, e.g. "5.67Gb"',
    )
    parser.add_argument("input", nargs=1, type=str, help="Path to input dataset")
    parser.add_argument("output", nargs=1, type=str, help="Path to output dataset")
    args = parser.parse_args()

    if args.workers > 1:
        from dask.distributed import Client, LocalCluster

        client = Client(
            LocalCluster(n_workers=args.workers, memory_limit=args.mem_per_worker)
        )
        print(client.dashboard_link)

    out_ds = aris_1go(xr.open_zarr(args.input[0]).load().chunk(location=1))

    out_ds.chunk(location=-1).to_zarr(args.output[0], mode="w")


def extract_point_data(ds, locations):
    """
    Extract data for specific point locations from a dataset.

    For each location provided, selects the nearest grid point in the dataset and
    concatenates the results along a new 'location' dimension.

    :param ds: Input xarray.Dataset.
    :type ds: xr.Dataset
    :param locations: DataFrame with location names and coordinates.
    :type locations: pd.DataFrame
    :return: Dataset with data for each specified location.
    :rtype: xr.Dataset
    """
    return (
        xr.concat(
            [
                ds.sel({**loc}, method="nearest").assign_coords(location=name)
                for name, loc in locations.iterrows()
            ],
            "location",
        )
        .chunk(location=-1)
        .transpose("time", ...)
    )
