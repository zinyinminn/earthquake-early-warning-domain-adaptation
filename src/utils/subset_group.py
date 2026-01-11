import os, h5py, pandas as pd
from tqdm import tqdm
import numpy as np

CSV_FULL = r"D:\datasets\stead_full\merge.csv"
H5_FULL  = r"D:\datasets\stead_full\merge.hdf5"
OUT_DIR  = r"D:\datasets\stead_subset"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_SUB  = os.path.join(OUT_DIR, "subset.csv")
H5_SUB   = os.path.join(OUT_DIR, "subset.hdf5")

# Choose target count. ~180k
TARGET_TRACES = 180_000  

ID_COL    = "trace_name"
LABEL_COL = "trace_category"

print("Loading full CSV (this is ~360MB; OK for RAM with 16 GB)…")
df = pd.read_csv(CSV_FULL)

# Normalize labels
df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower()

# Try to balance earthquake vs noise if present
if set(df[LABEL_COL].unique()) >= {"earthquake", "noise"}:
    half = TARGET_TRACES // 2
    eq  = df[df[LABEL_COL] == "earthquake"]
    noi = df[df[LABEL_COL] == "noise"]
    pick = pd.concat([
        eq.sample(min(half, len(eq)), random_state=42),
        noi.sample(min(half, len(noi)), random_state=42)
    ]).sample(frac=1.0, random_state=43).reset_index(drop=True)
else:
    # If only one label in this archive, just sample TARGET_TRACES
    pick = df.sample(TARGET_TRACES, random_state=42).reset_index(drop=True)

print("Selected:", len(pick), "rows")

# Copy waveforms from HDF5
missing = 0
with h5py.File(H5_FULL, "r") as fin, h5py.File(H5_SUB, "w") as fout:
    grp_out = fout.create_group("data")  # keep same structure
    grp_in  = fin["data"]

    for _, row in tqdm(pick.iterrows(), total=len(pick), desc="Copying"):
        tid = str(row[ID_COL])
        if tid in grp_in:
            ds = grp_in[tid]
            grp_out.create_dataset(tid, data=ds[()], compression="gzip")
        else:
            missing += 1

print("Missing datasets (not found in HDF5):", missing)

print("Writing subset CSV:", CSV_SUB)
pick.to_csv(CSV_SUB, index=False)

size_gb = os.path.getsize(H5_SUB) / (1024**3)
print(f"DONE. Subset HDF5 ≈ {size_gb:.1f} GB at {H5_SUB}")

