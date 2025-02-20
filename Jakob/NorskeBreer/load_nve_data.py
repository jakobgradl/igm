import numpy as np
import pandas as pd
import json, os
from pyproj import Transformer


def load_nve_data(params, state, glacier): # enter GlacierID: FNN -> Nordre Folgefonna, HAJ -> Hardangerjokulen
    with open("/home/ubuntu/pvol/NorskeBreer/data_istykkelse_nve_v2-0_2018/data_istykkelse_NVE_v2.0_2018.csv", 'r', encoding='cp865') as file:
        df = pd.read_csv(file, sep=';') 
    # with pd.read_csv("~/pvol/NorskeBreer/data_istykkelse_nve_v2-0_2018/data_istykkelse_NVE_v2.0_2018.csv", sep=';', encoding='cp865') as df:
    mask = (df["BreID/kode"] == glacier)
    data = df[mask]

    with open(os.path.join(params.oggm_RGI_ID, "glacier_grid.json"), "r") as f:
        ff = json.load(f)
        proj = ff["proj"]

    transformer = Transformer.from_crs("epsg:4326", proj, always_xy=True)
    lon = data["Geo_ost"]
    lat = data["Geo_Nord"]

    xx, yy = transformer.transform(lon, lat)

    usurf_nve = data["Hoeyde"].astype('float32')
    thk_nve = data["Istykkelse"].astype('float32')

    # Rasterize thickness data
    thickness_gridded = (
        pd.DataFrame(
            {
                "col": np.floor((xx - np.min(state.x)) / (state.x[1] - state.x[0])).astype(int),
                "row": np.floor((yy - np.min(state.y)) / (state.y[1] - state.y[0])).astype(int),
                "thickness": thk_nve,
            }
        )
        .groupby(["row", "col"])["thickness"]
        .mean()
    )
    thkobs = np.full((state.y.shape[0], state.x.shape[0]), np.nan)
    thickness_gridded[thickness_gridded == 0] = np.nan
    thkobs[tuple(zip(*thickness_gridded.index))] = thickness_gridded

    # Rasterize surface elevation data
    surface_gridded = (
        pd.DataFrame(
            {
                "col": np.floor((xx - np.min(state.x)) / (state.x[1] - state.x[0])).astype(int),
                "row": np.floor((yy - np.min(state.y)) / (state.y[1] - state.y[0])).astype(int),
                "surface": usurf_nve,
            }
        )
        .groupby(["row", "col"])["surface"]
        .mean()
    )
    usurfobs = np.full((state.y.shape[0], state.x.shape[0]), np.nan)
    surface_gridded[surface_gridded == 0] = np.nan
    usurfobs[tuple(zip(*surface_gridded.index))] = surface_gridded

    return usurfobs, thkobs



def load_aster_data(params):
    with open("/home/ubuntu/pvol/NorskeBreer/time_series_08/dh_08_rgi60_pergla_cumul.csv/dh_08_rgi60_pergla_cumul.csv", 'r', encoding='utf8') as file:
        df = pd.read_csv(file, sep=',') 
    # with pd.read_csv("~/pvol/NorskeBreer/data_istykkelse_nve_v2-0_2018/data_istykkelse_NVE_v2.0_2018.csv", sep=';', encoding='cp865') as df:
   
    pattern = '|'.join(params.RGIv6_basins)
    mask_basin = df["rgiid"].str.contains(pattern, regex=True)
    mask_time = df["time"].str.contains('-05-')
   
    data = df[mask_basin & mask_time]

    dS = pd.DataFrame(
        {"column": data['time'],
         "dS": data['dh'].astype('float32')*data['area'].astype('float32'),
         "area": data['area']}
    ).groupby("column").mean()

    dS = dS - dS.iloc[0]

    return dS