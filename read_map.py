import os
import pickle
import networkx as nx
from pyrosm import OSM


def cache_regional_osm_network():
    # Load the map file from the same directory as this script
    pbf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baden-wuerttemberg-260911.osm.pbf")

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Case_study")
    cache_file = os.path.join(cache_dir, "regional_network_cache_baden.pkl")

    print("=" * 70)
    print("[PREPROCESSING] Reading the original OSM PBF map file...")
    print("=" * 70)

    if not os.path.exists(pbf_file):
        print(f"✗ PBF file not found at path: {pbf_file}")
        return

    # Bounding box (Optional: adjust or remove if you want the entire file network)
    regional_box = [8.90, 48.50, 9.50, 49.00]

    osm = OSM(pbf_file, bounding_box=regional_box)

    print("  - Extracting the driving road network...")
    nodes, edges = osm.get_network(network_type="driving", nodes=True)

    print("  - Converting to an undirected NetworkX graph...")
    G_raw = osm.to_graph(nodes, edges, graph_type="networkx", simplify=True)
    G_base = G_raw.to_undirected()

    print(
        f"✓ Base road network loaded successfully: {G_base.number_of_nodes():,} nodes, {G_base.number_of_edges():,} edges.")

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    print(f"  - Saving cache to: {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(G_base, f)

    print("=" * 70)
    print("✓ MAP CACHING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    cache_regional_osm_network()
