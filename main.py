import pandas as pd
import networkx as nx
from collections import defaultdict
from difflib import get_close_matches
import heapq
import time
import math


CSV = "tfl_network_with_dlr.csv"
#interchanges add 3 mins
CHANGE_PENALTY = 3

#load station data
df = pd.read_csv(CSV)

required_cols = {"station1", "station2", "time_mins", "line"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

all_names = set(df["station1"]).union(set(df["station2"]))
canon_by_lower = {str(n).strip().lower(): str(n).strip() for n in all_names}
def canon(name: str) -> str:
    key = str(name).strip().lower()

    if key in canon_by_lower:
        return canon_by_lower[key]

    cands = get_close_matches(key, canon_by_lower.keys(), n=5, cutoff=0.75)
    if cands:
        suggestions = ", ".join(canon_by_lower[c] for c in cands)
        raise ValueError(f"Station '{name}' not found. Did you mean: {suggestions}?")

    raise ValueError(f"Station '{name}' not found.")

# build graph, nodes = (station,line), edges = (travel time)
# interchanges += 3 min penalty
G = nx.Graph()
lines_at_station = defaultdict(set)

for _, row in df.iterrows():
    a = str(row["station1"]).strip()
    b = str(row["station2"]).strip()
    t = float(row["time_mins"])
    line = str(row["line"]).strip()

    G.add_edge((a, line), (b, line), weight=t)
    lines_at_station[a].add(line)
    lines_at_station[b].add(line)

for station, lines in lines_at_station.items():
    lines = list(lines)
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            la, lb = lines[i], lines[j]
            G.add_edge((station, la), (station, lb), weight=CHANGE_PENALTY)

def variants(station_name: str):
    station_name = canon(station_name)
    return [(station_name, line) for line in lines_at_station[station_name]]


def build_query_graph(start_station: str, end_station: str):
    src = "__SRC__"
    dst = "__DST__"
    H = G.copy()

    for v in variants(start_station):
        H.add_edge(src, v, weight=0)

    for v in variants(end_station):
        H.add_edge(v, dst, weight=0)

    return H, src, dst


def strip_virtual_nodes(path):
    return [node for node in path if node not in ("__SRC__", "__DST__")]


def reconstruct_path(previous, src, dst):
    if src == dst:
        return [src]

    if dst not in previous:
        return []

    path = []
    current = dst

    while current != src:
        path.append(current)
        if current not in previous:
            return []
        current = previous[current]

    path.append(src)
    path.reverse()
    return path


def format_route_segments(path):
    if not path:
        return "No path found."

    if len(path) == 1:
        return f"{path[0][0]} [{path[0][1]}]"

    output = []
    seg_start = path[0][0]
    current_line = path[0][1]
    seg_time = 0

    for i in range(1, len(path)):
        prev_station, prev_line = path[i - 1]
        this_station, this_line = path[i]

        edge_weight = G[path[i - 1]][path[i]]["weight"]

        if prev_line == this_line:
            seg_time += edge_weight
        else:
            output.append(f"{seg_start} -> {prev_station} ({current_line}, {seg_time:.0f} min)")
            output.append(f"Change at {prev_station} (+{CHANGE_PENALTY} min)")
            seg_start = prev_station
            current_line = this_line
            seg_time = 0

    output.append(f"{seg_start} -> {path[-1][0]} ({current_line}, {seg_time:.0f} min)")
    return "\n".join(output)

# preprocessing (A* and Frankenalgorithm)
# landmark-based heuristic values
LANDMARK_STATIONS = [
    "Baker Street",
    "Bank",
    "King's Cross St. Pancras",
    "Oxford Circus",
    "Stratford",
    "Victoria"
]

landmark_distances = {}
def choose_landmark_node(station_name):
    vs = variants(station_name)
    if not vs:
        raise ValueError(f"No variants found for landmark {station_name}")
    return vs[0]
def preprocess_landmarks():
    global landmark_distances
    landmark_distances = {}

    for station in LANDMARK_STATIONS:
        try:
            node = choose_landmark_node(station)
            distances = nx.single_source_dijkstra_path_length(G, node, weight="weight")
            landmark_distances[node] = distances
        except Exception as e:
            print(f"Skipping landmark {station}: {e}")


def landmark_heuristic(node, target_variants):
    """
    Admissible lower bound:
    h(u,t) = max over landmarks L of |d(L,t) - d(L,u)|

    For destination with multiple line variants, use the minimum
    landmark distance to any destination variant.
    """
    if node in ("__SRC__", "__DST__"):
        return 0

    best_lb = 0

    for landmark_node, dist_map in landmark_distances.items():
        if node not in dist_map:
            continue

        d_l_u = dist_map[node]

        target_dists = [dist_map[t] for t in target_variants if t in dist_map]
        if not target_dists:
            continue

        d_l_t = min(target_dists)
        lb = abs(d_l_t - d_l_u)

        if lb > best_lb:
            best_lb = lb

    return best_lb

# Dijkstra's algorithm
def run_dijkstra(start_station, end_station):
    H, src, dst = build_query_graph(start_station, end_station)

    start_time = time.perf_counter()

    distances = {src: 0}
    previous = {}
    visited = set()
    counter = 0
    pq = [(0, counter, src)]
    nodes_explored = 0

    while pq:
        current_dist, _, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1

        if current == dst:
            break

        for neighbor in H.neighbors(current):
            weight = H[current][neighbor]["weight"]
            new_dist = current_dist + weight

            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current
                counter += 1
                heapq.heappush(pq, (new_dist, counter, neighbor))

    end_time = time.perf_counter()

    if dst not in distances:
        return [], math.inf, end_time - start_time, nodes_explored

    path = reconstruct_path(previous, src, dst)
    return strip_virtual_nodes(path), distances[dst], end_time - start_time, nodes_explored

# Bidirectional Dijkstra's
def run_bidirectional_dijkstra(start_station, end_station):
    H, src, dst = build_query_graph(start_station, end_station)

    start_time = time.perf_counter()

    dist_f = {src: 0}
    dist_b = {dst: 0}
    prev_f = {}
    prev_b = {}
    visited_f = set()
    visited_b = set()

    counter_f = 0
    counter_b = 0
    pq_f = [(0, counter_f, src)]
    pq_b = [(0, counter_b, dst)]

    best_meeting = None
    best_distance = math.inf
    nodes_explored = 0

    while pq_f and pq_b:
        if pq_f:
            d_f, _, u = heapq.heappop(pq_f)

            if u not in visited_f:
                visited_f.add(u)
                nodes_explored += 1

                if u in dist_b:
                    candidate = dist_f[u] + dist_b[u]
                    if candidate < best_distance:
                        best_distance = candidate
                        best_meeting = u

                for v in H.neighbors(u):
                    w = H[u][v]["weight"]
                    nd = dist_f[u] + w

                    if v not in dist_f or nd < dist_f[v]:
                        dist_f[v] = nd
                        prev_f[v] = u
                        counter_f += 1
                        heapq.heappush(pq_f, (nd, counter_f, v))

        if pq_b:
            d_b, _, u = heapq.heappop(pq_b)

            if u not in visited_b:
                visited_b.add(u)
                nodes_explored += 1

                if u in dist_f:
                    candidate = dist_f[u] + dist_b[u]
                    if candidate < best_distance:
                        best_distance = candidate
                        best_meeting = u

                for v in H.neighbors(u):
                    w = H[u][v]["weight"]
                    nd = dist_b[u] + w

                    if v not in dist_b or nd < dist_b[v]:
                        dist_b[v] = nd
                        prev_b[v] = u
                        counter_b += 1
                        heapq.heappush(pq_b, (nd, counter_b, v))

        min_f = pq_f[0][0] if pq_f else math.inf
        min_b = pq_b[0][0] if pq_b else math.inf

        if min_f + min_b >= best_distance:
            break

    end_time = time.perf_counter()

    if best_meeting is None:
        return [], math.inf, end_time - start_time, nodes_explored

    path_forward = reconstruct_path(prev_f, src, best_meeting)
    if not path_forward:
        return [], math.inf, end_time - start_time, nodes_explored

    backward_part = []
    current = best_meeting
    while current != dst:
        if current not in prev_b:
            return [], math.inf, end_time - start_time, nodes_explored
        current = prev_b[current]
        backward_part.append(current)

    full_path = path_forward + backward_part
    return strip_virtual_nodes(full_path), best_distance, end_time - start_time, nodes_explored

# Floyd-Warshall algorithm
def run_floyd_warshall(start_station, end_station):
    H, src, dst = build_query_graph(start_station, end_station)

    start_time = time.perf_counter()
    pred, dist = nx.floyd_warshall_predecessor_and_distance(H, weight="weight")
    end_time = time.perf_counter()

    if src not in dist or dst not in dist[src]:
        return [], math.inf, end_time - start_time, len(H.nodes)

    path = nx.reconstruct_path(src, dst, pred)
    total_cost = dist[src][dst]
    nodes_explored = len(H.nodes)

    return strip_virtual_nodes(path), total_cost, end_time - start_time, nodes_explored

# A* (using landmark heuristic)
def run_astar(start_station, end_station):
    H, src, dst = build_query_graph(start_station, end_station)
    end_variants = variants(end_station)

    start_time = time.perf_counter()

    g_score = {src: 0}
    previous = {}
    visited = set()
    counter = 0
    open_set = [(0, counter, src)]
    nodes_explored = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1

        if current == dst:
            break

        for neighbor in H.neighbors(current):
            tentative_g = g_score[current] + H[current][neighbor]["weight"]

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                previous[neighbor] = current
                h = landmark_heuristic(neighbor, end_variants)
                f = tentative_g + h
                counter += 1
                heapq.heappush(open_set, (f, counter, neighbor))

    end_time = time.perf_counter()

    if dst not in g_score:
        return [], math.inf, end_time - start_time, nodes_explored

    path = reconstruct_path(previous, src, dst)
    return strip_virtual_nodes(path), g_score[dst], end_time - start_time, nodes_explored

# Frankenalgorithm
#  Dijkstra's shortest path logic
# Bidirectional search
#  A* heuristic guidance
#  landmark preprocessing inspired by Floyd-Warshall
# cheaper preprocessing

def run_frankenalgorithm(start_station, end_station):
    H, src, dst = build_query_graph(start_station, end_station)

    start_variants = variants(start_station)
    end_variants = variants(end_station)

    start_time = time.perf_counter()

    g_f = {src: 0}
    g_b = {dst: 0}
    prev_f = {}
    prev_b = {}
    visited_f = set()
    visited_b = set()

    pq_f = []
    pq_b = []
    counter = 0

    heapq.heappush(pq_f, (0, counter, src))
    counter += 1
    heapq.heappush(pq_b, (0, counter, dst))
    counter += 1

    best_distance = math.inf
    best_meeting = None
    nodes_explored = 0

    def forward_h(node):
        return landmark_heuristic(node, end_variants)

    def backward_h(node):
        return landmark_heuristic(node, start_variants)

    while pq_f and pq_b:
        if pq_f:
            _, _, u = heapq.heappop(pq_f)

            if u not in visited_f:
                visited_f.add(u)
                nodes_explored += 1

                if u in g_b:
                    candidate = g_f[u] + g_b[u]
                    if candidate < best_distance:
                        best_distance = candidate
                        best_meeting = u

                for v in H.neighbors(u):
                    w = H[u][v]["weight"]
                    tentative_g = g_f[u] + w

                    if v not in g_f or tentative_g < g_f[v]:
                        g_f[v] = tentative_g
                        prev_f[v] = u
                        f_score = tentative_g + forward_h(v)
                        counter += 1
                        heapq.heappush(pq_f, (f_score, counter, v))

        if pq_b:
            _, _, u = heapq.heappop(pq_b)

            if u not in visited_b:
                visited_b.add(u)
                nodes_explored += 1

                if u in g_f:
                    candidate = g_f[u] + g_b[u]
                    if candidate < best_distance:
                        best_distance = candidate
                        best_meeting = u

                for v in H.neighbors(u):
                    w = H[u][v]["weight"]
                    tentative_g = g_b[u] + w

                    if v not in g_b or tentative_g < g_b[v]:
                        g_b[v] = tentative_g
                        prev_b[v] = u
                        f_score = tentative_g + backward_h(v)
                        counter += 1
                        heapq.heappush(pq_b, (f_score, counter, v))

        min_f = pq_f[0][0] if pq_f else math.inf
        min_b = pq_b[0][0] if pq_b else math.inf

        if min_f + min_b >= best_distance:
            break

    end_time = time.perf_counter()

    if best_meeting is None:
        return [], math.inf, end_time - start_time, nodes_explored

    path_forward = reconstruct_path(prev_f, src, best_meeting)
    if not path_forward:
        return [], math.inf, end_time - start_time, nodes_explored

    backward_part = []
    current = best_meeting
    while current != dst:
        if current not in prev_b:
            return [], math.inf, end_time - start_time, nodes_explored
        current = prev_b[current]
        backward_part.append(current)

    full_path = path_forward + backward_part
    return strip_virtual_nodes(full_path), best_distance, end_time - start_time, nodes_explored