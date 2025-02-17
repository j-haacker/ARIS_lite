import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

ctrl_dict = dict(
    evapotranspiration="ETA",
    Kc_factor="kc",
    max_air_temp="Tmax",
    min_air_temp="Tmin",
)

ctrl_point_grid = gpd.read_file("../data/reference/ARIS/small_grid_ids_500m")
ctrl_point_grid.index = pd.MultiIndex.from_frame(ctrl_point_grid[["LON_ID", "LAT_ID"]],
                                                 names=["LON_Index", "LAT_Index"])


def read_x_y_from_ctrl_grid(xix, yix):
    return ctrl_point_grid.loc[(xix, yix), ["x", "y"]].values


locations = pd.read_csv("../data/reference/ARIS/lonlat_inca_cropshift.csv", sep=";", header=0,
                        index_col="ID")
locations = locations.transform(lambda row: read_x_y_from_ctrl_grid(row.LON_Index, row.LAT_Index),
                                axis=1).rename(columns={"LON_Index": "x", "LAT_Index": "y"})


def read_ctrl_data(variable):
    df = pd.read_csv(f"../data/reference/ARIS/INCA_{ctrl_dict[variable]}_2020.csv", sep=";",
                     header=0, decimal=",")
    df.index = pd.MultiIndex.from_arrays([df["ID"], pd.to_datetime(df[["Year", "Month", "Day"]])],
                                         names=["ID", "date"])
    return df.loc(0)[:, "2020"].iloc(1)[-1]


def postprocess(da):
    if da.name in ["evapotranspiration"]:
        return da.sum("layer")
    return da


def compare_ctrl(da):
    ctrl_series = read_ctrl_data(da.name)
    for ID in ctrl_series.index.levels[0]:
        da.sel(**locations.loc[ID], method="nearest").pipe(postprocess).plot()
        ctrl_series.loc[ID].plot(ax=plt.gca())
        plt.show()
        