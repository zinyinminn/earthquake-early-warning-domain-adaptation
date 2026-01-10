# build_usgs_reg_h5.py
#
# Build HDF5 of spectrograms for USGS regression dataset
#   Input:
#       C:\Users\USER\eew_demo\data\usgs_big_reg\usgs_regression.csv
#       C:\Users\USER\eew_demo\data\usgs_big_reg\eq\USGS_EQ_*.mseed
#   Output:
#       C:\Users\USER\eew_demo\data\usgs_reg.hdf5
#       (same CSV, just cleaned if needed)

import os
import h5py
import numpy as np
import pandas as pd
from obspy import read

from spec_utils import make_spec_from_waveform   # use your existing implementation

# ---- paths (adjust if you moved things) -------------------------------------

BASE_DIR   = r"C:\Users\USER\eew_demo\data\usgs_big_reg"
CSV_IN     = os.path.join(BASE_DIR, "usgs_regression.csv")
MSEED_EQ   = os.path.join(BASE_DIR, "eq")
H5_OUT     = os.path.join(r"C:\Users\USER\eew_demo\data", "usgs_reg.hdf5")
CSV_OUT    = CSV_IN   # we reuse the same CSV

os.makedirs(os.path.dirname(H5_OUT), exist_ok=True)

FS = 100  # sampling rate your spectrograms expect

# ---- helper -----------------------------------------------------------------

def load_mseed(path):
    st = read(path)
    tr = st.merge(fill_value="interpolate")[0]
    data = tr.data.astype(np.float32)
    # ensure (3, T) format expected by make_spec_from_waveform
    if data.ndim == 1:
        x = np.stack([data, data, data], axis=0)
    else:
        x = data
    return x

# ---- main -------------------------------------------------------------------

def main():
    if not os.path.exists(CSV_IN):
        raise FileNotFoundError(f"CSV not found: {CSV_IN}")
    if not os.path.isdir(MSEED_EQ):
        raise FileNotFoundError(f"EQ folder not found: {MSEED_EQ}")

    df = pd.read_csv(CSV_IN)
    print(f"[INFO] usgs_regression.csv rows: {len(df)}")

    # verify we have trace_name column
    if "trace_name" not in df.columns:
        raise ValueError("CSV must contain a 'trace_name' column.")

    # open HDF5 to write
    with h5py.File(H5_OUT, "w") as h5:
        kept = 0
        for idx, row in df.iterrows():
            tn = str(row["trace_name"])
            mseed_path = os.path.join(MSEED_EQ, tn + ".mseed")
            if not os.path.exists(mseed_path):
                print(f"[WARN] missing waveform for {tn}, skipping row {idx}")
                continue

            try:
                wf = load_mseed(mseed_path)

                # *** IMPORTANT CHANGE ***
                # Call make_spec_from_waveform *without* the 'sampling_rate=' keyword
                # so we don't need to change spec_utils.py.
                spec = make_spec_from_waveform(wf, FS)
                # If your signature is make_spec_from_waveform(wf) only,
                # then just do: spec = make_spec_from_waveform(wf)

                spec = np.asarray(spec, dtype=np.float32)
                # ensure shape (3, 128, 128) or (C, H, W)
                if spec.ndim == 2:
                    spec = spec[None, ...]  # (1,H,W)
                h5.create_dataset(tn, data=spec, compression="gzip")
                kept += 1

                if kept % 100 == 0:
                    print(f"[INFO] saved {kept} spectrograms so far...")

            except Exception as e:
                print(f"[ERROR] Failed {tn}: {e}")

        print(f"[DONE] Wrote {kept} spectrograms to {H5_OUT}")

    print("[DONE] CSV kept as:", CSV_OUT)


if __name__ == "__main__":
    main()
