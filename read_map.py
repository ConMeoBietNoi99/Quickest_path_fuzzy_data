"""
Script tiền xử lý: Đọc file OpenStreetMap .pbf một lần và lưu cache dạng pickle.
Giúp các lần chạy case_study sau load cực nhanh mà không phải đọc lại file PBF.
"""

import os
import pickle
import networkx as nx
from pyrosm import OSM


def cache_regional_osm_network():
    pbf_file = r"D:\Research\The puickest path fuzzy\Code\Case_study\vietnam-260924.osm.pbf"
    #pbf_file = r"D:\Research\The puickest path fuzzy\Code\Case_study\luxembourg.pbf"
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Case_study")
    cache_file = os.path.join(cache_dir, "regional_network_cache_vietnam.pkl")

    print("=" * 70)
    print("[TIỀN XỬ LÝ] Đang đọc file bản đồ PBF gốc...")
    print("=" * 70)

    if not os.path.exists(pbf_file):
        print(f"✗ Không tìm thấy file PBF tại đường dẫn: {pbf_file}")
        return

    # Bounding box vùng liên đô thị (Stuttgart và vệ tinh)
    regional_box = [8.90, 48.50, 9.50, 49.00]

    osm = OSM(pbf_file, bounding_box=regional_box)

    print("  - Đang trích xuất mạng lưới đường giao thông (driving network)...")
    nodes, edges = osm.get_network(network_type="driving", nodes=True)

    print("  - Đang chuyển đổi thành đồ thị NetworkX vô hướng...")
    G_raw = osm.to_graph(nodes, edges, graph_type="networkx", simplify=True)
    G_base = G_raw.to_undirected()

    print(f"✓ Đã tải xong lưới đường gốc: {G_base.number_of_nodes():,} nút, {G_base.number_of_edges():,} cạnh.")

    # Lưu cache
    os.makedirs(cache_dir, exist_ok=True)
    print(f"  - Đang lưu cache vào: {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(G_base, f)

    print("=" * 70)
    print("✓ HOÀN TẤT LƯU CACHE BẢN ĐỒ THÀNH CÔNG!")
    print("=" * 70)


if __name__ == "__main__":
    cache_regional_osm_network()