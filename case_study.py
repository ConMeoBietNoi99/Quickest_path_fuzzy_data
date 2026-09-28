"""
Main script: Load OpenStreetMap network from local Cache (.pkl)
- Loads cached regional network instantly (skips slow PBF parsing)
- Allows interactive input of max_nodes and V_demand from keyboard
- Computes fuzzy quickest paths and plots results matching standard style
"""

import os
import pickle
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patheffects as pe


from qpp_fuzzy_solver import (
    TriangularFuzzy,
    Network,
    Arc,
    level_solver,parametric_algorithm
)

random.seed(42)

import os
import pickle
import random

import networkx as nx

# ============================================================================
# DATA FOR GENERATING FUZZY
# ============================================================================
ROAD_CHARACTERISTICS = {
    'motorway':       {'speed_kmh': 100, 'base_capacity': 35, 'congestion': 0.4, 'width_lanes': 4,   'urban': False},
    'trunk':          {'speed_kmh': 80,  'base_capacity': 28, 'congestion': 0.6, 'width_lanes': 4,   'urban': False},
    'primary':        {'speed_kmh': 70,  'base_capacity': 25, 'congestion': 0.7, 'width_lanes': 4,   'urban': False},
    'primary_link':   {'speed_kmh': 60,  'base_capacity': 20, 'congestion': 0.8, 'width_lanes': 3,   'urban': False},
    'secondary':      {'speed_kmh': 50,  'base_capacity': 18, 'congestion': 0.9, 'width_lanes': 2,   'urban': False},
    'secondary_link': {'speed_kmh': 45,  'base_capacity': 15, 'congestion': 1.0, 'width_lanes': 2,   'urban': False},
    'tertiary':       {'speed_kmh': 40,  'base_capacity': 12, 'congestion': 1.0, 'width_lanes': 2,   'urban': False},
    'residential':    {'speed_kmh': 30,  'base_capacity': 10, 'congestion': 1.2, 'width_lanes': 1.5, 'urban': True},
    'living_street':  {'speed_kmh': 20,  'base_capacity': 6,  'congestion': 1.3, 'width_lanes': 1,   'urban': True},
    'unclassified':   {'speed_kmh': 35,  'base_capacity': 8,  'congestion': 1.1, 'width_lanes': 1.5, 'urban': True},
    'service':        {'speed_kmh': 15,  'base_capacity': 4,  'congestion': 0.8, 'width_lanes': 1,   'urban': True},
}


def generate_realistic_fuzzy_params(length_m: float, highway_type: str):
    """Generate (lead_time, capacity) as  Triangular Fuzzy, based of type of road"""
    length_m = max(10.0, length_m)
    p = ROAD_CHARACTERISTICS.get(highway_type, ROAD_CHARACTERISTICS['residential'])

    base_speed = p['speed_kmh'] / 3.6
    nominal_time = length_m / base_speed
    cg = p['congestion']
    is_urban = p['urban']

    if is_urban:
        t_opt = nominal_time * random.uniform(0.50, 0.70)
        t_mid = nominal_time * random.uniform(1.00, 1.20)
        t_pes = nominal_time * (1.15 + cg * random.uniform(0.9, 1.9))
    else:
        t_opt = nominal_time * random.uniform(0.75, 0.90)
        t_mid = nominal_time * random.uniform(0.95, 1.15)
        t_pes = nominal_time * (1.05 + cg * random.uniform(0.35, 0.85))

    nominal_cap = p['base_capacity'] * p['width_lanes']
    local_quality = random.uniform(0.65, 1.35)

    if is_urban:
        c_pes = nominal_cap * local_quality * random.uniform(0.30, 0.50) / cg
        c_mid = nominal_cap * local_quality * random.uniform(0.60, 0.80)
        c_opt = nominal_cap * local_quality * random.uniform(0.90, 1.30)
    else:
        c_pes = nominal_cap * local_quality * random.uniform(0.55, 0.80) / cg
        c_mid = nominal_cap * local_quality * random.uniform(0.85, 1.00)
        c_opt = nominal_cap * local_quality * random.uniform(1.00, 1.20)

    dist_factor = 1.0 / (1.0 + (length_m / 1000) ** 0.3)
    c_pes *= dist_factor
    c_mid *= dist_factor
    c_opt *= dist_factor

    if c_pes >= c_mid:
        c_pes = c_mid * 0.8
    if c_mid >= c_opt:
        c_mid = (c_pes + c_opt) / 2
    if t_pes < t_mid:
        t_pes = t_mid * 1.2
    if t_mid < t_opt:
        t_opt = t_mid * 0.8

    return TriangularFuzzy(t_opt, t_mid, t_pes), TriangularFuzzy(c_pes, c_mid, c_opt)

def prepare_city_network_from_cache(max_nodes=None):
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Case_study", "regional_network_cache_baden.pkl")

    print("=" * 70)
    print(f"[1/4] Uploading the network from cache ...")
    print("=" * 70)

    if not os.path.exists(cache_path):
        print(f"✗ Cache file not found: {cache_path}")
        print("  -> Please run script 'read_map.py' to create cache first!")
        exit(1)

    with open(cache_path, "rb") as f:
        G_base = pickle.load(f)

    print(
        f"✓ Map loaded successfully: {G_base.number_of_nodes():,} nodes, {G_base.number_of_edges():,} edges.")

    print("[2/4] Analyze the connected structure...")
    if not nx.is_connected(G_base):
        largest_cc = max(nx.connected_components(G_base), key=len)
        G_active = G_base.subgraph(largest_cc).copy()
    else:
        G_active = G_base

    # Use BFS to take exactly connected max_nodes
    if max_nodes is not None and max_nodes > 0 and max_nodes < G_active.number_of_nodes():
        print(f"  -> Generating network with max_nodes = {max_nodes}...")
        degrees = dict(G_active.degree())
        center_node = max(degrees, key=degrees.get)

        bfs_nodes = []
        queue = [center_node]
        visited = {center_node}

        while queue and len(bfs_nodes) < max_nodes:
            curr = queue.pop(0)
            bfs_nodes.append(curr)
            neighbors = sorted(
                (nb for nb in G_active.neighbors(curr) if nb not in visited),
                key=lambda nb: degrees.get(nb, 0),
                reverse=True,
            )
            for neighbor in neighbors:
                visited.add(neighbor)
                queue.append(neighbor)

        G_active = G_active.subgraph(bfs_nodes).copy()
        n_junctions_kept = sum(1 for n in bfs_nodes if degrees.get(n, 0) >= 3)
        print(
            f"  -> Generated network with: {G_active.number_of_nodes():,} nodes, {G_active.number_of_edges():,} edges ")
    # =========================================================================
    # CHOSE SINK AND SOURCE
    # =========================================================================
    print("  -> Chosing sink and source...")
    nodes_list = list(G_active.nodes())
    source_node, sink_node = None, None
    source_name, sink_name = "Origin Terminal", "Destination Terminal"

    if len(nodes_list) > 1:
        valid_nodes = [n for n in nodes_list if 'x' in G_active.nodes[n] and 'y' in G_active.nodes[n]]
        if len(valid_nodes) < 2:
            valid_nodes = nodes_list

        named_nodes_pool = []
        for n in valid_nodes:
            node_data = G_active.nodes[n]
            if 'name' in node_data and node_data['name']:
                name_val = node_data['name']
                name_str = name_val[0] if isinstance(name_val, list) else str(name_val)
                named_nodes_pool.append((n, name_str))
                continue

            for neighbor in G_active.neighbors(n):
                edge_data = G_active.get_edge_data(n, neighbor)
                if edge_data:
                    found_road = False
                    for k, data in edge_data.items():
                        if 'name' in data and data['name']:
                            names = data['name']
                            road_name = names[0] if isinstance(names, list) else str(names)
                            named_nodes_pool.append((n, f"Junction: {road_name}"))
                            found_road = True
                            break
                    if found_road:
                        break

        if len(named_nodes_pool) >= 2:
            unique_named = {}
            for n, name in named_nodes_pool:
                if n not in unique_named:
                    unique_named[n] = name
            named_items = list(unique_named.items())

            max_dist = -1
            best_pair = (named_items[0][0], named_items[1][0])
            for i in range(len(named_items)):
                for j in range(i + 1, len(named_items)):
                    n1, name1 = named_items[i]
                    n2, name2 = named_items[j]
                    x1, y1 = G_active.nodes[n1]['x'], G_active.nodes[n1]['y']
                    x2, y2 = G_active.nodes[n2]['x'], G_active.nodes[n2]['y']
                    d = (x1 - x2) ** 2 + (y1 - y2) ** 2
                    if d > max_dist:
                        max_dist = d
                        best_pair = (n1, n2)
                        source_name = name1
                        sink_name = name2

            source_node, sink_node = best_pair

        if source_node is None or sink_node is None or not nx.has_path(G_active, source_node, sink_node):
            min_x = min(G_active.nodes[n]['x'] for n in valid_nodes)
            max_x = max(G_active.nodes[n]['x'] for n in valid_nodes)
            min_y = min(G_active.nodes[n]['y'] for n in valid_nodes)
            max_y = max(G_active.nodes[n]['y'] for n in valid_nodes)

            corner_NW = (min_x, max_y)
            corner_SE = (max_x, min_y)
            corner_SW = (min_x, min_y)
            corner_NE = (max_x, max_y)

            def find_closest(coord):
                return max(valid_nodes, key=lambda n: (G_active.nodes[n]['x'] - coord[0]) ** 2 + (
                            G_active.nodes[n]['y'] - coord[1]) ** 2)

            node_NW = find_closest(corner_NW)
            node_SE = find_closest(corner_SE)
            node_SW = find_closest(corner_SW)
            node_NE = find_closest(corner_NE)

            if ((G_active.nodes[node_NW]['x'] - G_active.nodes[node_SE]['x']) ** 2 +
                (G_active.nodes[node_NW]['y'] - G_active.nodes[node_SE]['y']) ** 2) >= \
                    ((G_active.nodes[node_SW]['x'] - G_active.nodes[node_NE]['x']) ** 2 +
                     (G_active.nodes[node_SW]['y'] - G_active.nodes[node_NE]['y']) ** 2):
                source_node, sink_node = node_NW, node_SE
            else:
                source_node, sink_node = node_SW, node_NE

            def get_street_name_near(node_id):
                for neighbor in G_active.neighbors(node_id):
                    edge_data = G_active.get_edge_data(node_id, neighbor)
                    if edge_data:
                        for k, data in edge_data.items():
                            if 'name' in data and data['name']:
                                names = data['name']
                                return names[0] if isinstance(names, list) else str(names)
                return None

            road_s = get_street_name_near(source_node)
            road_t = get_street_name_near(sink_node)

            source_name = f"Region Hub ({road_s})" if road_s else f"Origin Zone ({source_node})"
            sink_name = f"Terminal ({road_t})" if road_t else f"Destination Zone ({sink_node})"
    else:
        source_node = nodes_list[0]
        sink_node = nodes_list[0]

    print(f"✓ Source Location: {source_name} (Node ID: {source_node})")
    print(f"✓ Sink Location:   {sink_name} (Node ID: {sink_node})")

    print("[3/4] Creating DAG (Directed Acyclic Graph)...")
    lengths = nx.single_source_dijkstra_path_length(G_active, source_node, weight='length')

    G_dag = nx.DiGraph()
    for u, v, data in G_active.edges(data=True):
        if u == v:
            continue

        length_m = data.get('length', 100.0)
        d_u = lengths.get(u, float('inf'))
        d_v = lengths.get(v, float('inf'))

        if d_u < d_v:
            G_dag.add_edge(u, v, length=length_m)
        elif d_v < d_u:
            G_dag.add_edge(v, u, length=length_m)
        else:
            try:
                tie_break = u < v
            except TypeError:
                tie_break = str(u) < str(v)
            if tie_break:
                G_dag.add_edge(u, v, length=length_m)
            else:
                G_dag.add_edge(v, u, length=length_m)

    for node in G_dag.nodes():
        if node in G_active.nodes():
            G_dag.nodes[node]['x'] = G_active.nodes[node]['x']
            G_dag.nodes[node]['y'] = G_active.nodes[node]['y']

    topo_order = list(nx.topological_sort(G_dag))
    node_to_id = {node: idx for idx, node in enumerate(topo_order)}
    id_to_node = {idx: node for node, idx in node_to_id.items()}

    new_source = node_to_id[source_node]
    new_sink = node_to_id[sink_node]
    n_nodes = len(node_to_id)

    network = Network(n=n_nodes, source=new_source, sink=new_sink)
    added_count = 0

    print("[4/4] Generating fuzzy triangular numbers...")
    sample_arcs = []

    for u, v, data in G_dag.edges(data=True):
        if u not in node_to_id or v not in node_to_id:
            continue
        ui, vi = node_to_id[u], node_to_id[v]
        if ui >= vi:
            continue

        length_m = data.get('length', 100.0)
        highway_type = data.get('highway')
        if not highway_type:
            if length_m > 8000:
                highway_type = 'motorway'
            elif length_m > 2000:
                highway_type = 'primary'
            elif length_m > 500:
                highway_type = 'secondary'
            else:
                highway_type = 'residential'

        lead_time, capacity = generate_realistic_fuzzy_params(length_m, highway_type)
        arc = Arc(i=ui, j=vi, lead_time=lead_time, capacity=capacity)
        network.add_arc(arc)

        if len(sample_arcs) < 5:
            sample_arcs.append((ui, vi, length_m, lead_time, capacity))

        added_count += 1

    print(f"✓ Network created successfully: {n_nodes} nodes, {added_count} edges.\n")

    return network, G_active, id_to_node, new_source, new_sink, source_name, sink_name

def get_node_name_label(G, node_id, id_to_node, default_prefix="Origin"):
    actual_node = id_to_node.get(node_id, node_id)
    node_data = G.nodes.get(actual_node, {})

    # 1. Kiểm tra tên trực tiếp trên node
    if 'name' in node_data and node_data['name']:
        return str(node_data['name'])

    # 2. Quét qua các cạnh kề để tìm tên đường / tên địa danh (neighborhood search)
    for neighbor in G.neighbors(actual_node):
        edge_data = G.get_edge_data(actual_node, neighbor)
        if edge_data:
            # Trường hợp đồ thị đa cạnh (MultiDiGraph hoặc MultiGraph)
            if isinstance(edge_data, dict):
                for key, data in edge_data.items():
                    if 'name' in data and data['name']:
                        names = data['name']
                        return names[0] if isinstance(names, list) else str(names)

    # 3. Nếu vẫn không có tên, trả về tọa độ hoặc nhãn mô tả thân thiện hơn
    if 'x' in node_data and 'y' in node_data:
        return f"Location ({node_data['x']:.2f}, {node_data['y']:.2f})"

    return f"{default_prefix} (ID: {actual_node})"

def plot_original_map(G_active, output_dir, max_nodes=None):
    if not G_active:
        return None

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_maxnodes_{max_nodes}" if max_nodes is not None else "_full"

    print("  -> Generating and exporting detailed original map...")
    fig_raw, ax_raw = plt.subplots(figsize=(16, 14), facecolor='white')
    n_nodes = G_active.number_of_nodes()
    opt_k = 10.0 / math.sqrt(n_nodes) if n_nodes > 0 else None
    pos_raw = nx.spring_layout(G_active, k=opt_k, scale=6.0, iterations=1000, weight=None, seed=42)
    nx.draw_networkx_edges(G_active, pos_raw, edge_color="#222222", width=2.2, alpha=0.9, ax=ax_raw)
    nx.draw_networkx_nodes(G_active, pos_raw, node_size=140, node_color="#1f77b4", edgecolors="white", linewidths=2.0,
                           ax=ax_raw)

    node_count_str = f"({n_nodes} nodes, {G_active.number_of_edges()} edges)"
    ax_raw.set_title(f"Original Road Network Map {node_count_str}", fontsize=18, fontweight='bold', pad=25)
    ax_raw.axis('off')

    raw_png_path = os.path.join(output_dir, f"original_network_map{suffix}.png")
    raw_pdf_path = os.path.join(output_dir, f"original_network_map{suffix}.pdf")

    fig_raw.savefig(raw_png_path, dpi=300, bbox_inches='tight')
    fig_raw.savefig(raw_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig_raw)

    print(f"  ✓ Saved original map PNG: {raw_png_path}")
    print(f"  ✓ Saved original map PDF: {raw_pdf_path}")

    return pos_raw

def _edge_key(u_node, v_node):
    try:
        return (u_node, v_node) if u_node <= v_node else (v_node, u_node)
    except TypeError:
        return tuple(sorted((u_node, v_node), key=str))


def _build_edge_usage(paths_dict_by_alpha, id_to_node):
    edge_usage = {}
    alphas = sorted(paths_dict_by_alpha.keys())
    for alpha in alphas:
        for path_idx, path in enumerate(paths_dict_by_alpha[alpha]):
            optimal_node_ids = [id_to_node[n_idx] for n_idx in path.nodes if n_idx in id_to_node]
            for k in range(len(optimal_node_ids) - 1):
                ek = _edge_key(optimal_node_ids[k], optimal_node_ids[k + 1])
                edge_usage.setdefault(ek, []).append((alpha, path_idx))
    return edge_usage


def _draw_paths_with_shared_edge_offset(ax, pos, paths_dict_by_alpha, id_to_node,
                                        edge_usage, color_map, alphas,
                                        spacing_step, main_width, sub_width):
    num_alphas = len(alphas)
    for idx, alpha in enumerate(alphas):
        paths_list = paths_dict_by_alpha[alpha]
        if not paths_list:
            continue
        color = color_map(idx / max(1, num_alphas - 1))

        for path_idx, path in enumerate(paths_list):
            optimal_node_ids = [id_to_node[n_idx] for n_idx in path.nodes if n_idx in id_to_node]
            if len(optimal_node_ids) < 2:
                continue

            for k in range(len(optimal_node_ids) - 1):
                u_node = optimal_node_ids[k]
                v_node = optimal_node_ids[k + 1]
                if u_node not in pos or v_node not in pos:
                    continue
                ek = _edge_key(u_node, v_node)
                sharers = edge_usage[ek]
                pos_in_edge = sharers.index((alpha, path_idx))
                n_sharers = len(sharers)
                shift_factor = (pos_in_edge - (n_sharers - 1) / 2.0) * spacing_step
                p1, p2 = np.array(pos[u_node]), np.array(pos[v_node])
                direction = p2 - p1
                length = np.linalg.norm(direction)
                if length == 0:
                    continue
                normal = np.array([-direction[1], direction[0]]) / length
                s_p1 = p1 + normal * shift_factor
                s_p2 = p2 + normal * shift_factor

                ax.plot([s_p1[0], s_p2[0]], [s_p1[1], s_p2[1]], color=color,
                        linewidth=main_width if path_idx == 0 else sub_width,
                        alpha=0.95, solid_capstyle='round', zorder=3)


def _declutter_layout(pos, min_dist_frac=0.6, iterations=300):
    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes], dtype=float)
    n = len(nodes)
    if n < 2:
        return pos

    dmins = []
    for i in range(n):
        d = np.linalg.norm(coords - coords[i], axis=1)
        d[i] = np.inf
        dmins.append(d.min())
    min_dist = float(np.median(dmins)) * (1.0 + min_dist_frac)

    try:
        from scipy.spatial import cKDTree
        use_scipy = True
    except ImportError:
        use_scipy = False

    for _ in range(iterations):
        if use_scipy:
            tree = cKDTree(coords)
            pairs = list(tree.query_pairs(min_dist))
        else:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                     if np.linalg.norm(coords[i] - coords[j]) < min_dist]
        if not pairs:
            break
        disp = np.zeros_like(coords)
        for i, j in pairs:
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-9:
                diff = np.random.randn(2) * 1e-3
                dist = np.linalg.norm(diff)
            push = (min_dist - dist) / 2.0 * (diff / dist)
            disp[i] += push
            disp[j] -= push
        coords += disp * 0.5 

    return {node: tuple(coords[i]) for i, node in enumerate(nodes)}


def _auto_pos_raw(G_active):
    n_nodes = G_active.number_of_nodes()
    if n_nodes == 0:
        return {}

    has_coords = all('x' in G_active.nodes[n] and 'y' in G_active.nodes[n] for n in G_active.nodes())
    init_pos = {n: (G_active.nodes[n]['x'], G_active.nodes[n]['y']) for n in G_active.nodes()} if has_coords else None

    try:
        if n_nodes > 800:
            raise RuntimeError("The graph is too large for Kamada-Kawai, switching to spring_layout")
        pos = nx.kamada_kawai_layout(G_active, weight=None, pos=init_pos, scale=8.0)
    except Exception:
        opt_k = 12.0 / math.sqrt(n_nodes)
        pos = nx.spring_layout(G_active, k=opt_k, scale=8.0, iterations=1200,
                               weight=None, seed=42, pos=init_pos)

    return _declutter_layout(pos, min_dist_frac=0.6, iterations=300)


def _auto_spacing_step(pos, fraction=0.006):
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if not xs:
        return 0.01
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return max(diag * fraction, 1e-9)

def create_grid_layout(G):
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    ncols = max(1, round(np.sqrt(n * 1.4)))
    nrows = int(np.ceil(n / ncols))

    pos = {}
    for idx, node in enumerate(nodes):
        r = idx // ncols  
        c = idx % ncols  

        x = c * 2.0
        y = -r * 2.0  

        pos[node] = np.array([x, y])

    return pos



def plot_optimal_paths(paths_dict_by_alpha, G_active, id_to_node, output_dir, V,
                       algorithm_tag, pos_raw=None, max_nodes=None,
                       source_node=None, sink_node=None):
    if not G_active or not paths_dict_by_alpha:
        return

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_maxnodes_{max_nodes}" if max_nodes is not None else "_full"

    title_tag = "Parametric Algorithm" if algorithm_tag == "parametric" else "Level Solver"
    file_tag = "parametric" if algorithm_tag == "parametric" else "level_solver"

    if pos_raw is None:
        pos_raw = _auto_pos_raw(G_active)

    source_name = get_node_name_label(G_active, source_node, id_to_node, default_prefix="Origin")
    sink_name = get_node_name_label(G_active, sink_node, id_to_node, default_prefix="Destination")

    edge_width_bg = 2.5
    opt_path_width_main = edge_width_bg * 2.0
    opt_path_width_sub = edge_width_bg * 1.4

    alphas = sorted(paths_dict_by_alpha.keys())
    color_map = mpl.colormaps.get_cmap('plasma').resampled(len(alphas))
    num_alphas = len(alphas)

    edge_usage = _build_edge_usage(paths_dict_by_alpha, id_to_node)
    n_shared_edges = sum(1 for v in edge_usage.values() if len(v) > 1)

    # =========================================================================
    # IMAGE 1: OPTIMAL PATH ON GRID (Grid Layout)
    # =========================================================================
    print(f"  -> Generating optimal paths map (Grid Layout - {title_tag})...")
    fig_opt, ax_opt = plt.subplots(figsize=(16, 14), facecolor='white')

    pos_grid = create_grid_layout(G_active)

    resolved_source = source_node
    if resolved_source not in pos_grid and resolved_source in id_to_node:
        resolved_source = id_to_node[resolved_source]

    resolved_sink = sink_node
    if resolved_sink not in pos_grid and resolved_sink in id_to_node:
        resolved_sink = id_to_node[resolved_sink]

    nx.draw_networkx_edges(G_active, pos_grid, edge_color="#888888", width=edge_width_bg, alpha=0.5, ax=ax_opt)

    skip_nodes = {n for n in [resolved_source, resolved_sink, source_node, sink_node] if n is not None}
    other_nodes = [n for n in G_active.nodes() if n not in skip_nodes]

    nx.draw_networkx_nodes(G_active, pos_grid, nodelist=other_nodes, node_size=200, node_color="#1f77b4",
                           edgecolors="white", linewidths=2.5, ax=ax_opt)

    if resolved_source is not None and resolved_source in pos_grid:
        coords = pos_grid[resolved_source]
        ax_opt.scatter([coords[0]], [coords[1]], s=500, marker='s', color='#2ecc71', edgecolors='black', linewidths=3.0,
                       zorder=5)
        ax_opt.text(coords[0], coords[1] + 0.18, f"Source: {source_name}", fontsize=12, fontweight='bold',
                    color="#27ae60", ha='center', va='bottom', zorder=6,
                    path_effects=[pe.withStroke(linewidth=4, foreground="white")])

    if resolved_sink is not None and resolved_sink in pos_grid:
        coords = pos_grid[resolved_sink]
        ax_opt.scatter([coords[0]], [coords[1]], s=550, marker='^', color='#e74c3c', edgecolors='black', linewidths=3.0,
                       zorder=5)
        ax_opt.text(coords[0], coords[1] + 0.18, f"Sink: {sink_name}", fontsize=12, fontweight='bold', color="#c0392b",
                    ha='center', va='bottom', zorder=6,
                    path_effects=[pe.withStroke(linewidth=4, foreground="white")])

    alpha_spacing_step = _auto_spacing_step(pos_grid, fraction=0.02)

    _draw_paths_with_shared_edge_offset(
        ax_opt, pos_grid, paths_dict_by_alpha, id_to_node, edge_usage,
        color_map, alphas, alpha_spacing_step, opt_path_width_main, opt_path_width_sub,
    )

    legend_elements = [
        mlines.Line2D([], [], color='none', marker='s', markersize=11, markerfacecolor='#2ecc71',
                      markeredgecolor='black', label=f'Source: {source_name}'),
        mlines.Line2D([], [], color='none', marker='^', markersize=12, markerfacecolor='#e74c3c',
                      markeredgecolor='black', label=f'Sink: {sink_name}'),
        mlines.Line2D([], [], color='none', label='--- Confidence Levels ---', alpha=0)
    ]
    legend_elements.extend(
        [mlines.Line2D([], [], color=color_map(i / max(1, num_alphas - 1)), lw=opt_path_width_main, label=f'α = {alpha:.2f}') for
         i, alpha in enumerate(alphas)])

    ax_opt.legend(handles=legend_elements, loc='upper right', fontsize=11, title="Legend & Parameters",
                  title_fontsize=12, frameon=True)
    ax_opt.set_title(
        f"Optimal Quickest Inter-City Paths ({title_tag}, Grid Layout, V = {V}, Max Nodes = {max_nodes if max_nodes else 'Full'})",
        fontsize=18, fontweight='bold', pad=25)
    ax_opt.axis('off')

    opt_png_path = os.path.join(output_dir, f"inter_city_quickest_paths_{file_tag}{suffix}.png")
    opt_pdf_path = os.path.join(output_dir, f"inter_city_quickest_paths_{file_tag}{suffix}.pdf")
    fig_opt.savefig(opt_png_path, dpi=300, bbox_inches='tight')
    fig_opt.savefig(opt_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig_opt)

    print(f"  ✓ Saved grid paths map PNG: {opt_png_path}")
    print(f"  ✓ Saved grid paths map PDF: {opt_pdf_path}")

    # =========================================================================
    # IMAGE 2: OPTIMAL PATH ON ORIGINAL MAP
    # =========================================================================
    print(f"  -> Generating original map with optimal paths ({title_tag})...")
    fig_orig, ax_orig = plt.subplots(figsize=(16, 14), facecolor='white')

    nx.draw_networkx_edges(G_active, pos_raw, edge_color="#222222", width=edge_width_bg, alpha=0.75, ax=ax_orig)
    nx.draw_networkx_nodes(G_active, pos_raw, nodelist=other_nodes, node_size=200, node_color="#1f77b4",
                           edgecolors="white", linewidths=2.5, ax=ax_orig)

    if resolved_source is not None and resolved_source in pos_raw:
        coords = pos_raw[resolved_source]
        ax_orig.scatter([coords[0]], [coords[1]], s=500, marker='s', color='#2ecc71', edgecolors='black',
                        linewidths=3.0, zorder=5)
        ax_orig.text(coords[0], coords[1] + 0.04, f"Source: {source_name}", fontsize=12, fontweight='bold',
                     color="#27ae60", ha='center', va='bottom', zorder=6,
                     path_effects=[pe.withStroke(linewidth=4, foreground="white")])

    if resolved_sink is not None and resolved_sink in pos_raw:
        coords = pos_raw[resolved_sink]
        ax_orig.scatter([coords[0]], [coords[1]], s=550, marker='^', color='#e74c3c', edgecolors='black',
                        linewidths=3.0, zorder=5)
        ax_orig.text(coords[0], coords[1] + 0.04, f"Sink: {sink_name}", fontsize=12, fontweight='bold', color="#c0392b",
                     ha='center', va='bottom', zorder=6,
                     path_effects=[pe.withStroke(linewidth=4, foreground="white")])

    alpha_spacing_step_orig = _auto_spacing_step(pos_raw, fraction=0.006)

    _draw_paths_with_shared_edge_offset(
        ax_orig, pos_raw, paths_dict_by_alpha, id_to_node, edge_usage,
        color_map, alphas, alpha_spacing_step_orig, opt_path_width_main, opt_path_width_sub,
    )

    ax_orig.legend(handles=legend_elements, loc='upper right', fontsize=11,
                   title_fontsize=12, frameon=True)
    ax_orig.set_title(
        f"Optimal Quickest Inter-City Paths (Original Map Layout, {title_tag}, V = {V})",
        fontsize=18, fontweight='bold', pad=25)
    ax_orig.axis('off')

    orig_png_path = os.path.join(output_dir, f"original_map_with_{file_tag}_paths{suffix}.png")
    orig_pdf_path = os.path.join(output_dir, f"original_map_with_{file_tag}_paths{suffix}.pdf")
    fig_orig.savefig(orig_png_path, dpi=300, bbox_inches='tight')
    fig_orig.savefig(orig_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig_orig)

    print(f"  ✓ Saved original map with paths PNG: {orig_png_path}")
    print(f"  ✓ Saved original map with paths PDF: {orig_pdf_path}\n")

def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Case_study")

    print("=" * 70)
    print("INITIAL PARAMETER CONFIGURATION")
    print("=" * 70)

    try:
        max_node_input = input(
            ">> Enter max_nodes (Press Enter or type 'None' to use default full region): ").strip()
        if max_node_input == "" or max_node_input.lower() == "none":
            max_nodes = None
            print("-> Selected: Using default node count in the region.")
        else:
            max_nodes = int(max_node_input)
            print(f"-> Selected: Limited max_nodes = {max_nodes}")
    except ValueError:
        print("! Invalid value, automatically using default (None).")
        max_nodes = None

    print()

    network, G_active, id_to_node, source_idx, sink_idx, auto_source_name, auto_sink_name = prepare_city_network_from_cache(
        max_nodes=max_nodes)

    print("=" * 70)
    print("TRANSPORTATION DEMAND SETTING")
    print("=" * 70)
    ALPHA_LEVELS = [0.25, 0.5, 0.75]
    user_input_v = input('Enter value for V (Press Enter to keep automatic value): ').strip()
    if user_input_v != "":
        try:
            V_demand = float(user_input_v)
        except ValueError:
            print("! Invalid value, keeping automatic V_demand.")

    print("=" * 70)
    print("RENDERING ORIGINAL MAP")
    print("=" * 70 + "\n")

    pos_raw = plot_original_map(G_active, output_dir, max_nodes=max_nodes)

    print("=" * 70)
    print(f"RUNNING LEVEL SOLVER ALGORITHM (V = {V_demand})")
    print("=" * 70 + "\n")

    all_paths_by_alpha = {}
    for alpha in ALPHA_LEVELS:
        print(f"Processing confidence level α = {alpha}...")
        level_result = level_solver(network, alpha, V_demand)

        if not level_result.paths:
            print(f"  ✗ No paths found at α = {alpha}\n")
            continue

        all_paths_by_alpha[alpha] = level_result.paths
        print(f"   ✓ Found {len(level_result.paths)} optimal paths.\n")

    print("=" * 70)
    print("RENDERING INTER-CITY NETWORK RESULTS (LEVEL SOLVER)")
    print("=" * 70 + "\n")

    plot_optimal_paths(
        all_paths_by_alpha, G_active, id_to_node, output_dir, V_demand,
        algorithm_tag="level_solver",
        pos_raw=pos_raw,
        max_nodes=max_nodes,
        source_node=source_idx,
        sink_node=sink_idx
    )

    print("=" * 70)
    print(f"RUNNING PARAMETRIC ALGORITHM (V = {V_demand})")
    print("=" * 70 + "\n")

    print("  -> Computing critical alpha partitions in [0, 1]...")
    parametric_result = parametric_algorithm(network, V=V_demand, max_refine_rounds=4)

    total_solves = getattr(parametric_result, 'total_bsp_solves', getattr(parametric_result, 'bsp_solves', 'N/A'))
    print(f"  ✓ Parametric partition completed! (Total BSP solves = {total_solves})")

    parametric_paths_by_alpha = {}
    raw_cells = None
    for attr_name in ['merged', 'critical_cells', 'cells', 'intervals', 'partition']:
        if hasattr(parametric_result, attr_name):
            raw_cells = getattr(parametric_result, attr_name)
            break

    if raw_cells is None and isinstance(parametric_result, (list, tuple, dict)):
        raw_cells = parametric_result

    if raw_cells:
        if isinstance(raw_cells, dict):
            parametric_paths_by_alpha = raw_cells
        else:
            for item in raw_cells:
                try:
                    if len(item) == 2:
                        bounds, paths_list = item
                        if paths_list:
                            if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                                alpha_repr = (float(bounds[0]) + float(bounds[1])) / 2.0
                            else:
                                alpha_repr = float(bounds)
                            parametric_paths_by_alpha[alpha_repr] = paths_list
                except Exception:
                    continue

    if parametric_paths_by_alpha:
        print(f"  -> Total extracted alpha partition intervals: {len(parametric_paths_by_alpha)}")
        print("  -> Exporting result maps for Parametric Algorithm...")

        plot_optimal_paths(
            paths_dict_by_alpha=parametric_paths_by_alpha,
            G_active=G_active,
            id_to_node=id_to_node,
            output_dir=output_dir,
            V=V_demand,
            algorithm_tag="parametric",
            pos_raw=pos_raw,
            max_nodes=max_nodes,
            source_node=source_idx,
            sink_node=sink_idx
        )
    else:
        print("! Warning: Could not map data from ParametricResult to alpha dict.")

    print("=" * 70)
    print("ALL PROCESSES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
