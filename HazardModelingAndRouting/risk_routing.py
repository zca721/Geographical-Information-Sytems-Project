# risk_routing.py
"""
Risk-Aware Routing Module
Dev: Lamya
-------------------------

This file implements our algorithmic contribution:
    - A composite cost function balancing travel time + flood risk
    - Risk-aware shortest path routing using Dijkstra
    - Ability to vary alpha (risk sensitivity)
    - Reporting of time/risk tradeoffs

Inputs:
    - outputs/road_network_with_risk.graphml  (created by hazard_model.py)

Outputs:
    - Path results for different alpha values
    - Time and risk metrics per route
"""

import networkx as nx
import osmnx as ox
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl


def ensure_numeric_risk(G):
    for u, v, k, data in G.edges(keys=True, data=True):
        try:
            data["risk"] = float(data.get("risk", 0.0))
        except:
            data["risk"] = 0.0
    return G


# 1. Loading Graph

def load_graph_with_risk(graph_path="outputs/road_network_with_risk.graphml"):
    """
    Loads the graph after hazard_model.py added 'risk' attribute.
    """
    print("Loading graph with risk...")
    G = ox.load_graphml(graph_path)
    G = ensure_numeric_risk(G) 
    
    # Project the graph to UTM meters
    G = ox.project_graph(G)
    # ensure travel_time exists
    for u, v, k, data in G.edges(keys=True, data=True):
        if "travel_time" not in data:
            # default speed 40 km/h = 11.11 m/s
            length = data.get("length", 1)
            data["travel_time"] = length / 11.11

    return G



# 2. Composite Cost Function

def apply_composite_cost(
    G,
    alpha: float,
    time_key="travel_time",
    risk_key="risk",
    cost_key="cost"
):
    """
    Defining composite cost:
        cost = travel_time + alpha * risk

    alpha = 0   => pure shortest time
    alpha > 0   => risk-aware routing
    """
    print(f"Applying composite cost with alpha = {alpha} ...")

    for u, v, k, data in G.edges(keys=True, data=True):
        t = float(data.get(time_key, 0.0))
        r = float(data.get(risk_key, 0.0))
        data[cost_key] = t + alpha * r

    return G


# 3. Find Nearest Network Node

def nearest_node(G, x, y):
    """
    Given projected coordinates (x,y) return nearest graph node
    """
    return ox.distance.nearest_nodes(G, X=x, Y=y)


# 4. Run Risk-Aware Shortest Path

def compute_route(G, source_node: int, target_node: int, alpha, cost_key="cost") -> List[int]:
    """
    Compute shortest path using Dijkstra with custom cost
    """
    print("Computing shortest path...")
    path = nx.shortest_path(G, source=source_node, target=target_node, weight=cost_key)
    plot_route(G, path, alpha, save_path=f"outputs/route_alpha_{alpha}.png")
    return path
    #plotting

def plot_route(G, path, alpha, save_path=None):
    """
    Plot the route with risk-based color coding.
    Blue = low risk
    Red = high risk
    """

    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw the full graph in light gray
    ox.plot_graph(G, ax=ax, node_size=0, edge_color="#cccccc", show=False, close=False)

    # Extract risk values for each road segment in the chosen route
    route_edges = list(zip(path[:-1], path[1:]))

    for u, v in route_edges:
        data = list(G.get_edge_data(u, v).values())[0]

        geom = data.get("geometry")
        risk = data.get("risk", 0.0)

        # Color map from blue → red depending on risk
        cmap = plt.cm.jet
        color = cmap(risk)

        if geom is not None:
            xs, ys = geom.xy
            ax.plot(xs, ys, color=color, linewidth=3)
        else:
            # fallback if no geometry is stored
            x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
            x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)

    plt.title(f"Route with α = {alpha} (risk-aware)")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved color-coded route → {save_path}")

    plt.close()


# 5. Evaluate Path Time and Risk
def path_time_and_risk(
    G,
    path: List[int],
    time_key="travel_time",
    risk_key="risk"
) -> Tuple[float, float]:
    """
    Sum travel time and risk along the route
    """
    total_time = 0.0
    total_risk = 0.0

    for u, v in zip(path[:-1], path[1:]):
        data = list(G.get_edge_data(u, v).values())[0]

        # TIME 
        time_val = data.get(time_key, 0.0)
        try:
            time_val = float(time_val)
        except:
            time_val = 0.0
        total_time += time_val

        # RISK 
        risk_val = data.get(risk_key, 0.0)
        try:
            risk_val = float(risk_val)
        except:
            risk_val = 0.0
        total_risk += risk_val

    return total_time, total_risk


# 6. Experimenting Multiple Alpha Values

def run_routing_experiment(
    graph_path: str,
    origin_xy: Tuple[float, float],
    dest_xy: Tuple[float, float],
    alpha_values: List[float]
):
    """
    High-level function:
        - Load graph with risk
        - Find nearest nodes
        - For each alpha: apply cost, compute route, compute metrics
        - Return all results for plotting/reporting
    """

    # Load graph with risk
    G = load_graph_with_risk(graph_path)

    # Convert coordinates to nearest nodes
    print("Finding nearest nodes...")
    src_node = nearest_node(G, *origin_xy)
    dst_node = nearest_node(G, *dest_xy)
    print(f"Source node = {src_node}, Target node = {dst_node}")

    # Collect results
    results = []

    for alpha in alpha_values:
        print("\n==============================================")
        print(f"Running routing with alpha = {alpha}")

        G = apply_composite_cost(G, alpha)

        # Compute path
        path = compute_route(G, src_node, dst_node, alpha)

        # Compute metrics
        total_time, total_risk = path_time_and_risk(G, path)

        print(f"Route length (#nodes): {len(path)}")
        print(f"Total travel time (seconds): {total_time:.2f}")
        print(f"Total risk (sum): {total_risk:.4f}")

        results.append({
            "alpha": alpha,
            "path": path,
            "total_time": total_time,
            "total_risk": total_risk
        })

    return results



# 7. Script Entry Point (Optional)

if __name__ == "__main__":
    # Example coordinates (should be in the AOI zone)
    origin = (507931, 5206000)   # placeholder
    dest   = (509911, 5210400)   # placeholder
    

    alphas = [0, 20, 50, 100]

    results = run_routing_experiment(
        graph_path="outputs/road_network_with_risk.graphml",
        origin_xy=origin,
        dest_xy=dest,
        alpha_values=alphas
    )

    print("\nEXPERIMENT COMPLETE.")
