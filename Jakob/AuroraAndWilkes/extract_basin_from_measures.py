#!/usr/bin/env python3

import numpy as np
import xarray as xr
import geopandas as gp
# import rasterio as rast
from rasterio import features as rf
from rasterio import transform as rt

from rasterio.transform import from_origin
from rasterio.features import geometry_mask


basins = gp.read_file('~/pvol/measures_data/Basins_IMBIE_Antarctica_v02/Basins_IMBIE_Antarctica_v02.shp')
bm3 = xr.open_dataset('~/pvol/measures_data/BedMachineAntarctica-v3.nc')
uvel = xr.open_dataset('~/pvol/measures_data/antarctic_ice_vel_phase_map_v01.nc')

# Wilkes basin -> basins.iloc[5], Aurora basin -> basin.iloc[6]
basins = basins.iloc[5:7]

transform = from_origin(-3333000, 3333000, 500,500)
# (minx, maxy, res, res) from bm3

mask = geometry_mask(basins2.geometry, transform=transform, invert=True, out_shape=(13333,13333))
data_array = xr.DataArray(mask, dims=("y", "x"), coords={"y": np.arange(-3333000, 3333001, 500), "x": np.arange(-3333000, 3333001, 500)})

data_array.to_netcdf("mask_WilkesAurora_MEASURES.nc")





### get data for make_AW

import xarray as xr
import numpy as np

bm3 = xr.open_dataset('~/pvol/measures_data/BedMachineAntarctica-v3.nc')
maskWA = xr.open_dataset('~/pvol/measures_data/mask_WilkesAurora_MEASURES.nc')
maskWAshelf = xr.open_dataset('pvol/measures_data/mask_IceShelves_WilkesAuroraCROPPED_MEASURES.nc')

# bedmachine Wilkes and Aurora
bm3WAcrop = bm3.where(maskWA,drop=True)

# Bedmachine WA shelves
surfshelf = bm3['surface'].where(maskWAshelf)
bedshelf = bm3['bed'].where(maskWAshelf)
thkshelf = bm3['thickness'].where(maskWAshelf)
maskshelf = bm3['mask'].where(maskWAshelf)

WAshelf = xr.Dataset({
    "surface": surfshelf['__xarray_dataarray_variable__'],
    "bed": bedshelf['__xarray_dataarray_variable__'],
    "thickness": thkshelf['__xarray_dataarray_variable__'],
    "mask": maskshelf['__xarray_dataarray_variable__']
    })

WAshelf.to_netcdf('pvol/measures_data/BedMachine-v3-WilkesAurora-shelf.nc')

# add measures velocity
# make a mask for the shelf portions
