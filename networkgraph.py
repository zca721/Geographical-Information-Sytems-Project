"""
Network Graph Module
Dev: Sanya
Purpose:
    Build the road network graph from OpenStreetMap (OSM) data,
    compute travel times, and flag flooded edges using the
    flood extent polygons generated in the Hazard Modeling step.
"""

import os
import osmnx as ox
import geopandas as gpd

def build_network(place_name="Thurston County, Washington, USA"):
    print("Downloading OSM road network for:", place_name)
    # Get drivable road network
    G = ox.graph_from_place(place_name, network_type="drive")

    # Add a travel_time attribute (seconds) assuming 50 km/h if not specified
    for u, v, k, data in G.edges(keys=True, data=True):
        if "length" in data:
            data["travel_time"] = data["length"] / (50 * 1000 / 3600)

    print("Road network built with", len(G.edges), "edges.")
    return G

def flag_flooded_edges(G, flood_polygon_path="outputs/flood_extent.gpkg"):
    if not os.path.exists(flood_polygon_path):
        print("Flood extent file not found, skipping flood marking.")
        return G

    flood_zones = gpd.read_file(flood_polygon_path)
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)

    # Spatial join: identify edges intersecting flood polygons
    flooded = gpd.sjoin(edges_gdf, flood_zones, how="inner", predicate="intersects")
    flooded_ids = set(flooded.index)

    for u, v, k, data in G.edges(keys=True, data=True):
        data["flooded"] = (data.get("osmid") in flooded_ids)

    print(f"Flagged {len(flooded_ids)} flooded road segments.")
    return G

def export_graph(G, out_path="outputs/road_network.graphml"):
    ox.save_graphml(G, out_path)
    print("💾 Graph exported to:", out_path)

def run_network_graph():
    G = build_network()
    G = flag_flooded_edges(G)
    export_graph(G)

if __name__ == "__main__":
    run_network_graph()
