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