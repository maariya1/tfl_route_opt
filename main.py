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
