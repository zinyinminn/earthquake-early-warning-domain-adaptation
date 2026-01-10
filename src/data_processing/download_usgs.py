# build_cross_dataset.py
# Make a STEAD-like CSV + HDF5 from ANY catalog + waveform folder.
# Robust to missing fields: never raises, fills with NaN / "" as needed.

import os, glob
import numpy as np
import pandas as pd
import h5py
from obspy import read

# Your spectrogram helper
from dataloader_h5 import wf_to_spec  # returns (3,128,128) from (T,3) waveform

# ===================== CONFIG =====================
DATASET      = "USGS"
CATALOG      = r"D:\datasets\cross\USGS\catalog.csv"
WAVEFORM_DIR = r"D:\datasets\cross\USGS\waveforms"
OUT_CSV      = rf"D:\datasets\cross\{DATASET.lower()}_subset.csv"
OUT_H5       = rf"D:\datasets\cross\{DATASET.lower()}_subset.hdf5"
# ==================================================

FS, WIN_SEC, SPEC_SIZE = 100, 30, 128

# Final columns we’ll emit (STEAD-like)
OUT_COLS = [
    "trace_name", "label_eq",
    "sp_sec", "dist_km", "mag",
    "source_lat", "source_lon",
    "receiver_lat", "receiver_lon",
    "network", "station", "location",
    "file"
]

# Common synonyms we’ll try to map automatically
SYN = {
    "trace_name":  ["trace_name", "id", "evid", "event_id"],
    "label_eq":    ["label_eq", "label", "is_eq"],
    "mag":         ["mag", "magnitude", "Mw", "ml", "Ml", "mb", "Mb"],
    "source_lat":  ["source_lat", "event_lat", "evla", "latitude", "lat"],
    "source_lon":  ["source_lon", "event_lon", "evlo", "longitude", "lon"],
    "receiver_lat":["receiver_lat", "station_lat", "stla"],
    "receiver_lon":["receiver_lon", "station_lon", "stlo"],
    "network":     ["network", "net"],
    "station":     ["station", "sta"],
    "location":    ["location", "loc"],
    "file":        ["file", "filename", "path"]
    # sp_sec, dist_km usually not present in cross datasets → will become NaN
}

def pick(df: pd.DataFrame, keys, default=None):
    """Return first present column among synonyms, else default series/value."""
    for k in keys:
        if k in df.columns: 
            return df[k]
    if isinstance(default, (int, float, str)) or default is None:
        # broadcast scalar default to full-length series
        return pd.Series([default]*len(df))
    return default

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    out["trace_name"]   = pick(df, SYN["trace_name"], default=None).astype(str)
    # classifier label: if not present, assume earthquake=1 (we downloaded quakes)
    out["label_eq"]     = pick(df, SYN["label_eq"], default=1).fillna(1).astype(int)

    # regression targets: may be missing in cross datasets (leave NaN)
    out["sp_sec"]       = df["sp_sec"] if "sp_sec" in df.columns else np.nan
    out["dist_km"]      = df["dist_km"] if "dist_km" in df.columns else np.nan
    out["mag"]          = pd.to_numeric(pick(df, SYN["mag"], default=np.nan), errors="coerce")

    # coordinates (NaN if not available)
    out["source_lat"]   = pd.to_numeric(pick(df, SYN["source_lat"], default=np.nan), errors="coerce")
    out["source_lon"]   = pd.to_numeric(pick(df, SYN["source_lon"], default=np.nan), errors="coerce")
    out["receiver_lat"] = pd.to_numeric(pick(df, SYN["receiver_lat"], default=np.nan), errors="coerce")
    out["receiver_lon"] = pd.to_numeric(pick(df, SYN["receiver_lon"], default=np.nan), errors="coerce")

    # id/meta
    out["network"]      = pick(df, SYN["network"],  "").astype(str)
    out["station"]      = pick(df, SYN["station"],  "").astype(str)
    out["location"]     = pick(df, SYN["location"], "").astype(str)
    out["file"]         = pick(df, SYN["file"],     "").astype(str)

    return pd.DataFrame(out, columns=OUT_COLS)

def find_mseed(trace_name: str, filecol: str) -> str:
    """Find waveform path using file column or filename patterns."""
    if filecol:
        p = os.path.join(WAVEFORM_DIR, filecol)
        if os.path.isfile(p): 
            return p
    pats = [
        os.path.join(WAVEFORM_DIR, f"{trace_name}_*.mseed"),
        os.path.join(WAVEFORM_DIR, f"{trace_name}.mseed"),
        os.path.join(WAVEFORM_DIR, f"*{trace_name}*.mseed"),
    ]
    for p in pats:
        g = glob.glob(p)
        if g: return g[0]
    return ""

def preprocess(path: str) -> np.ndarray:
    """Load MiniSEED → 3ch, resample to FS, crop/pad to WIN_SEC."""
    st = read(path)
    st.merge(fill_value="interpolate")
    # pick first 3 channels; if fewer, pad zeros
    trs = st[:3]
    while len(trs) < 3:
        z = trs[0].copy(); z.data[:] = 0
        trs.append(z)
    for tr in trs:
        tr.resample(FS)
    T = FS * WIN_SEC
    def fixlen(x):
        if len(x) > T: return x[:T]
        if len(x) < T: return np.pad(x, (0, T-len(x)))
        return x
    wf = np.stack([fixlen(tr.data) for tr in trs], axis=1)  # (T,3)
    return wf.astype(np.float32)

def main():
    df_raw = pd.read_csv(CATALOG)
    df = ensure_schema(df_raw)

    if os.path.exists(OUT_H5): os.remove(OUT_H5)
    h5 = h5py.File(OUT_H5, "w")
    g = h5.create_group("data")

    rows = []
    ok, skip = 0, 0

    for _, row in df.iterrows():
        tn = str(row["trace_name"]) if pd.notna(row["trace_name"]) and str(row["trace_name"]).strip() else None
        # Make a fallback name if missing
        if not tn:
            tn = f"row{_}"
        path = find_mseed(tn, row["file"])
        if not path:
            skip += 1
            continue
        try:
            wf = preprocess(path)
            spec = wf_to_spec(wf, fs=FS, size=SPEC_SIZE).astype(np.float32)
            g.create_dataset(tn, data=spec, compression="gzip", compression_opts=4)
            out_row = [row[c] for c in OUT_COLS]
            out_row[OUT_COLS.index("trace_name")] = tn  # ensure name used in H5
            rows.append(out_row)
            ok += 1
        except Exception:
            skip += 1

    h5.close()
    pd.DataFrame(rows, columns=OUT_COLS).to_csv(OUT_CSV, index=False)
    print(f"Done. ok={ok}, skip={skip}")
    print(f"CSV → {OUT_CSV}")
    print(f"HDF5 → {OUT_H5}")

if __name__ == "__main__":
    main()
