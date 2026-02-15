# hazard_model.py
"""
Hazard Modeling Module (DEM-based Flood Susceptibility)
Dev: Lamya :) 
-------------------------------------------------------

This script creates a risk layer using DEM elevation:
    - Low elevation  => high flood risk
    - High elevation => low flood risk

Pipeline:
1. Load DEM raster.
2. Load road network graph (from networkgraph.py).
3. Sample DEM values along each road segment.
4. Convert DEM -> risk score.
5. Save updated graph with a 'risk' attribute.

This file MUST be run BEFORE risk_routing.py.
"""

import numpy as np
import networkx as nx
import osmnx as ox
import rasterio
from shapely.geometry import LineString


# Loading graph and DEM

def load_graph(graph_path="outputs/road_network.graphml"):
    """Loading graph created by sanya"""
    return ox.load_graphml(graph_path)


def load_dem_raster(dem_path="./rem-files/FullAreaOfRivers.tif"):
    """Loading DEM raster (GeoTIFF)."""
    src = rasterio.open(dem_path)
    arr = src.read(1, masked=True)
    return src, arr


# DEM sampling along edges

def sample_dem_along_line(line: LineString, src, n_samples=10):
    """
    Interpolate DEM values every few meters along a road segment.
    Returns an array of DEM values sampled along that line.
    """
    distances = np.linspace(0, line.length, n_samples)
    points = [line.interpolate(d) for d in distances]
    coords = [(p.x, p.y) for p in points]

    values = np.array([val[0] for val in src.sample(coords)], dtype=np.float32)
    values = np.where(values < -1e6, np.nan, values)  # removing invalid values

    return values


# Converting elevation to a flood risk score

def elevation_to_risk(dem_values, min_elev=None, max_elev=None):
    """
    DEM -> Risk conversion.
    lower elevation = higher flood risk.

    Formula we are using:
        risk = (max_elev - elevation) / (max_elev - min_elev)

    Output is normalized to [0,1].
    """
    valid = dem_values[~np.isnan(dem_values)]
    if valid.size == 0:
        return 0.0

    elev = np.mean(valid)  # using average elevation along the road

    # Global normalization based on DEM range
    if min_elev is None:
        min_elev = valid.min()
    if max_elev is None:
        max_elev = valid.max()

    risk = (max_elev - elev) / (max_elev - min_elev)
    return float(np.clip(risk, 0.0, 1.0))


# Assigning risk to edges


def assign_dem_risk_to_edges(
    G,
    dem_path="./rem-files/FullAreaOfRivers.tif",
    n_samples=10
):
    """
    For each road edge, we compute DEM-based risk.
    """

    src, arr = load_dem_raster(dem_path)

    global_min = np.nanmin(arr)
    global_max = np.nanmax(arr)

    print(f"DEM range: min={global_min:.2f}, max={global_max:.2f}")

    for u, v, k, data in G.edges(keys=True, data=True):

        geom = data.get("geometry", None)

        if geom is None or not isinstance(geom, LineString):
            data["risk"] = 0.0
            continue

        dem_values = sample_dem_along_line(geom, src, n_samples=n_samples)
        risk_score = elevation_to_risk(dem_values, global_min, global_max)

        data["risk"] = float(risk_score)

    src.close()
    return G


# Saving the updated graph


def save_graph(G, out_path="outputs/road_network_with_risk.graphml"):
    ox.save_graphml(G, out_path)
    print(f"Saved graph with DEM-based risk → {out_path}")



# Main driver

def run_dem_hazard_model():
    print("Loading graph...")
    G = load_graph()

    print("Assigning DEM-based flood risk...")
    G = assign_dem_risk_to_edges(G)

    print("Saving updated graph...")
    save_graph(G)

    print("Hazard modeling complete!")



if __name__ == "__main__":
    run_dem_hazard_model()
