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