from pathlib import Path
from IPython.core.display import Video

import numpy as np
import pandas as pd
import geopandas as gpd  # Vector data handling
import osmnx as ox       # Downloading data from OSM

from shapely.geometry import box
from scipy.spatial import cKDTree as KDTree # For Inverse Distance Weight calculation

import xarray as xr    
import xrspatial    # Hillshading
import rioxarray    # Working with geospatial data in xarray

import matplotlib.pyplot as plt
from datashader.transfer_functions import shade, stack

import geojson

import shapely

import matplotlib

# Displays a topographical map within coordinates selected
dem = rioxarray.open_rasterio('./rem-files/ThurtsonCounty.tif')

geom = '''{"type": "Polygon",
                "coordinates":[[
                [-123.0217381703,45.9961409058],
                [-122.0975705076,45.9961409058],
                [-122.0975705076,47.0431092798],
                [-123.0217381703,47.0431092798],
                [-123.0217381703,45.9961409058]]]}'''

cropping_geometries = [geojson.loads(geom)]
cropped = dem.rio.clip(geometries=cropping_geometries, crs=4326)

cropped = cropped.coarsen(x=3, boundary='trim').mean().coarsen(y=3, boundary='trim').mean()

cropped.squeeze().plot.imshow()



# Displays entire river that is specified
# Deschutes River
river = ox.geocode_to_gdf('Deschutes River, Washington', which_result=1)
river = river.to_crs(cropped.rio.crs)

river.plot()

# Cowlitz River
# river_1 = ox.geocode_to_gdf('Cowlitz River, Washington', which_result=1)
# river_1 = river_1.to_crs(cropped.rio.crs)

# river_1.plot()

# Nisqually River
# river_2 = ox.geocode_to_gdf('Nisqually River, Washington', which_result=1)
# river_2 = river_2.to_crs(cropped.rio.crs)

# river_2.plot()

# Displays topographical map with river combined
cropped.rio.bounds()

bounds = cropped.rio.bounds()
xmin, ymin, xmax, ymax = bounds

# Deschutes River
river = river.clip(bounds)

river_geom = river.geometry.iloc[0]
river_geom

# Cowlitz River
# river_1 = river_1.clip(bounds)

# river_geom_1 = river_1.geometry.iloc[0]
# river_geom_1

# Nisqually River
# river_2 = river_2.clip(bounds)

# river_geom_2 = river_2.geometry.iloc[0]
# river_geom_2

cropped = cropped.sel(y=slice(ymax, ymin), x=slice(xmin, xmax))

fig, ax = plt.subplots()
cropped.squeeze().plot.imshow(ax=ax)
# Deschutes River
river.plot(ax=ax, color='red')

# Cowlitz River
# river_1.plot(ax=ax, color='orange')

# Nisqually River
# river_2.plot(ax=ax, color='yellow')



# Smooth representation of different elevation on topographic map
def split_coords(geom):
    x = []
    y = []
    for i in shapely.get_coordinates(geom):
        x.append(i[0])
        y.append(i[1])
    return x, y

xs, ys = split_coords(river_geom)
xs, ys = xr.DataArray(xs, dims='z'), xr.DataArray(ys, dims='z')

sampled = cropped.interp(x=xs, y=ys, method='nearest').dropna(dim='z')

# Sampled river coordinates
c_sampled = np.vstack([sampled.coords[c].values for c in ('x', 'y')]).T

# All (x, y) coordinates of the original DEM
c_x, c_y = [cropped.coords[c].values for c in ('x', 'y')]
c_interpolate = np.dstack(np.meshgrid(c_x, c_y)).reshape(-1, 2)

# Sampled values
values = sampled.values.ravel()
c_interpolate

tree = KDTree(c_sampled)

# IWD interpolation
distances, indices = tree.query(c_interpolate, k=50) # k value changes smoothness

weights = 1 / distances
weights = weights / weights.sum(axis=1).reshape(-1, 1)

interpolated_values = (weights * values[indices]).sum(axis=1)
interpolated_values

elevation_raster = xr.DataArray(
    interpolated_values.reshape((len(c_y), len(c_x))).T, dims=('x', 'y'), coords={'x': c_x, 'y': c_y}
)

fig, ax = plt.subplots()
elevation_raster.transpose().plot.imshow(ax=ax)
# Deschutes River
river.plot(ax=ax, color='red')



# Final full representation of river levels
rem = cropped - elevation_raster

# colors = ['#FFFFFF', '#ADD8E6', '#6C3BAA']
colors = ['#f2f7fb', '#81a8cb', '#37123d']

shade(rem.squeeze(), cmap=colors, span=[0, 10], how='linear')

a = shade(xrspatial.hillshade(cropped.squeeze(), angle_altitude=1, azimuth=310), cmap=['black', 'white'], how='linear')
b = shade(rem.squeeze(), cmap=colors, span=[0, 10], how='linear', alpha=225) # Alpha values 0 and 225 for transparency
stack(a, b)

# matplotlib.pyplot.imshow()

# final_map = stack(a, b)

# Display the final map
# final_map.plot.imshow()

# Required for matplotlib to display graphs in Python outside of Jupyter Notebook
# Only need once to display all graphs
plt.show()

# fig, ax = plt.subplots(figsize=(10, 10))

# # Alternatively, to save the plot to a file instead of displaying
# fig.savefig('my_shaded_map.png', dpi=300, bbox_inches='tight')
# plt.close(fig) # Close the figure to free up memory