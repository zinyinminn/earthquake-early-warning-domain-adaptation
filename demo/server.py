import json
import threading
import time
import csv
import os
import random   # NEW: for realistic noise
from queue import Queue, Empty

from flask import Flask, render_template, Response, jsonify, request

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
ROOT = os.path.dirname(__file__)
CFG_PATH = os.path.join(ROOT, "config.json")

with open(CFG_PATH, "r") as f:
    CFG = json.load(f)

DATA_CFG = CFG["data"]
SERVER_CFG = CFG["server"]
MODELS_CFG = CFG.get("models", {})

# main STEAD manifest from config
STEAD_MANIFEST_CSV = os.path.join(ROOT, DATA_CFG["manifest_csv"])

# dataset CSV mapping (keys match dropdown + metrics_cross.json)
DATASETS = {
    "STEAD": os.path.join(ROOT, "data", "subset.csv"),
    "MYANMAR": os.path.join(ROOT, "data", "myanmar_demo.csv"),
    "USGS_real": os.path.join(ROOT, "data", "USGS_real.csv"),
}

WAVE_COL = DATA_CFG.get("manifest_wave_col", "trace_name")
MAG_COL_CONF = DATA_CFG.get("manifest_mag_col", "source_ma")
LAT_COL_CONF = DATA_CFG.get("manifest_lat_col", "source_lat")
LON_COL_CONF = DATA_CFG.get("manifest_lon_col", "source_lon")
P_COL = DATA_CFG.get("manifest_p_col", "p_arrival_sample")
S_COL = DATA_CFG.get("manifest_s_col", "s_arrival_sample")

SIM_INTERVAL = float(SERVER_CFG.get("sim_interval_sec", 1.5))
ALERT_MAG = float(SERVER_CFG.get("alert_mag_threshold", 3.0))
SAMPLING_HZ = float(SERVER_CFG.get("sampling_hz", 100.0))

HOST = SERVER_CFG.get("host", "0.0.0.0")
PORT = int(SERVER_CFG.get("port", 8000))

# ---------------------------------------------------------------------
# Load offline metrics once (for realistic noise)
# ---------------------------------------------------------------------
METRICS_CROSS = {}
METRICS_PATH = os.path.join(ROOT, "models", "metrics_cross.json")
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, "r") as f:
            METRICS_CROSS = json.load(f)
    except Exception as e:
        print("Warning: could not load metrics_cross.json:", e)
else:
    print("Warning: metrics_cross.json not found at", METRICS_PATH)

def get_regression_noise_scales(dataset_key: str):
    """
    Get noise scales (approximate MAE) for magnitude, distance and S-P time
    from metrics_cross.json. If missing, fall back to reasonable defaults.
    """
    ds = METRICS_CROSS.get(dataset_key, {})
    reg = ds.get("regression", {}) if isinstance(ds, dict) else {}

    try:
        mag = float(reg.get("mag_mae", 0.3) or 0.3)
    except Exception:
        mag = 0.3
    try:
        dist = float(reg.get("dist_mae_km", 20.0) or 20.0)
    except Exception:
        dist = 20.0
    try:
        sp = float(reg.get("sp_mae_s", 0.5) or 0.5)
    except Exception:
        sp = 0.5

    # We will use these as standard deviations for zero-mean noise.
    return mag, dist, sp

def get_classification_error_rate(dataset_key: str):
    """
    Use offline accuracy to approximate how often to flip predictions
    so that live accuracy looks similar to offline.
    """
    ds = METRICS_CROSS.get(dataset_key, {})
    clf = ds.get("classification", {}) if isinstance(ds, dict) else {}
    try:
        acc = float(clf.get("accuracy", 0.99) or 0.99)
    except Exception:
        acc = 0.99

    # error rate = 1 - accuracy, but clamp between 0 and 0.4
    err = 1.0 - acc
    err = max(0.0, min(0.4, err))
    return err

# ---------------------------------------------------------------------
# App + globals
# ---------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

stream_q = Queue(maxsize=64)
stop_event = threading.Event()
sim_thread = None

# manifest cache + pointers per dataset
MANIFEST_ROWS = {}   # dataset -> list of rows
CURRENT_INDEX = {}   # dataset -> next row index
CURRENT_DATASET = None  # which dataset is currently being streamed

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def read_manifest_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def _load_manifest_from_disk(dataset_key):
    """
    Load the CSV for a given dataset key.
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    path = DATASETS[dataset_key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest CSV not found for {dataset_key}: {path}")
    rows = read_manifest_csv(path)
    if not rows:
        raise RuntimeError(f"Manifest CSV for {dataset_key} loaded but contains 0 rows.")
    return rows

def get_manifest(dataset_key):
    """
    Load manifest once per dataset and cache it in MANIFEST_ROWS.
    """
    global MANIFEST_ROWS
    if dataset_key not in MANIFEST_ROWS:
        MANIFEST_ROWS[dataset_key] = _load_manifest_from_disk(dataset_key)
    return MANIFEST_ROWS[dataset_key]

def pick_column(row, preferred_names, fallback_contains=None):
    """
    Try preferred names first, then any column whose name contains a substring.
    """
    for name in preferred_names:
        if name in row and row[name] not in ("", None):
            return row[name]
    if fallback_contains:
        fl = fallback_contains.lower()
        for k, v in row.items():
            if fl in k.lower() and v not in ("", None):
                return v
    return None

def parse_label(raw):
    """
    Map dataset label to 0/1 (0 = noise, 1 = earthquake).
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s.startswith("earth"):
            return 1
        if s.startswith("noise") or s.startswith("noisy"):
            return 0
        try:
            v = int(float(s))
            return 1 if v > 0 else 0
        except Exception:
            return None
    try:
        v = int(float(raw))
        return 1 if v > 0 else 0
    except Exception:
        return None

# ---------------------------------------------------------------------
# "Inference" from CSV (demo but now more realistic)
# ---------------------------------------------------------------------
def infer_from_row(row, dataset_key: str):
    """
    Simulated model outputs using ground truth + noise to mimic
    realistic offline performance for each dataset.

    In a real system you would run your trained PyTorch model here,
    but this keeps the demo fast and visually meaningful.
    """
    # --- regression noise scales (per dataset) ---
    mag_sigma, dist_sigma, sp_sigma = get_regression_noise_scales(dataset_key)
    class_err = get_classification_error_rate(dataset_key)

    # --- true magnitude from dataset ---
    mag_true_raw = pick_column(
        row,
        [MAG_COL_CONF, "source_ma", "source_mag", "source_magnitude"],
        fallback_contains="mag",
    )
    try:
        mag_true = float(mag_true_raw or 0.0)
    except Exception:
        mag_true = 0.0

    # predicted magnitude = true + noise
    mag_pred = max(0.0, random.gauss(mag_true, mag_sigma))

    # --- true distance (km) from source_dis or similar ---
    dist_true_raw = pick_column(
        row, ["source_dis", "source_distance_km"], fallback_contains="dis"
    )
    try:
        if dist_true_raw is not None and dist_true_raw != "":
            d_val = float(dist_true_raw)
            # if very small, assume degrees
            dist_true_km = d_val * 111.0 if d_val < 5 else d_val
        else:
            dist_true_km = None
    except Exception:
        dist_true_km = None

    if dist_true_km is None:
        dist_true_km = 50.0  # fallback
    dist_pred_km = max(0.0, random.gauss(dist_true_km, dist_sigma))

    # --- S-P time (seconds) ---
    sp_true_sec = None
    try:
        p_samp = float(row.get(P_COL, "") or 0.0)
        s_samp = float(row.get(S_COL, "") or 0.0)
        if s_samp > p_samp and SAMPLING_HZ > 0:
            sp_true_sec = (s_samp - p_samp) / SAMPLING_HZ
    except Exception:
        sp_true_sec = None

    if sp_true_sec is not None:
        sp_pred_sec = max(0.0, random.gauss(sp_true_sec, sp_sigma))
    else:
        sp_pred_sec = None

    # --- classification label (earthquake vs noise) ---
    label_raw = pick_column(
        row, ["label_eq", "trace_category", "event_type"], "trace_cat"
    )
    true_label = parse_label(label_raw)

    pred_label = None
    if true_label is not None:
        # base classifier: based on predicted magnitude
        base_pred = 1 if mag_pred >= 1.5 else 0
        # flip some predictions to roughly match offline accuracy
        if random.random() < class_err:
            pred_label = 1 - base_pred
        else:
            pred_label = base_pred

    # ETA to user (very simple model)
    eta_sec = max(0.5, dist_pred_km / 6.0)

    # epicenter
    lat_raw = pick_column(row, [LAT_COL_CONF, "source_lat"], fallback_contains="lat")
    lon_raw = pick_column(row, [LON_COL_CONF, "source_lon"], fallback_contains="lon")
    try:
        lat = float(lat_raw or 0.0)
        lon = float(lon_raw or 0.0)
    except Exception:
        lat, lon = 0.0, 0.0

    epic = {"lat": lat, "lon": lon}

    # default user (frontend can override)
    user = {"lat": lat + 0.3, "lon": lon + 0.3}

    inference_ms = int(30 + random.random() * 40)

    return {
        # predictions
        "magnitude": mag_pred,
        "distance_km": dist_pred_km,
        "sp_pred_sec": sp_pred_sec,
        "pred_label": pred_label,

        # ground truth
        "true_mag": mag_true,
        "true_dist_km": dist_true_km,
        "sp_true_sec": sp_true_sec,
        "true_label": true_label,

        "eta_seconds": eta_sec,
        "epicenter": epic,
        "user": user,
        "inference_ms": inference_ms,
        "alert": mag_pred >= ALERT_MAG,
    }

# ---------------------------------------------------------------------
# Simulation loop  (per dataset)
# ---------------------------------------------------------------------
def sim_loop(dataset_key: str, start_idx: int):
    """
    Stream events for a given dataset starting from start_idx.
    Updates CURRENT_INDEX[dataset] so we can continue after Stop.
    """
    global CURRENT_INDEX

    rows = get_manifest(dataset_key)

    stop_event.clear()

    n_rows = len(rows)
    if start_idx >= n_rows:
        start_idx = 0
        CURRENT_INDEX[dataset_key] = 0

    for i in range(start_idx, n_rows):
        if stop_event.is_set():
            break

        row = rows[i]
        pred = infer_from_row(row, dataset_key)   # <-- pass dataset_key

        try:
            stream_q.put(pred, timeout=2)
        except Exception:
            pass

        # next index to send if we resume later
        CURRENT_INDEX[dataset_key] = i + 1

        # choose wait time
        wait = SIM_INTERVAL
        try:
            p = float(row.get(P_COL, "") or 0.0)
            s = float(row.get(S_COL, "") or 0.0)
            if s > p and SAMPLING_HZ > 0:
                sp = (s - p) / SAMPLING_HZ
                wait = max(0.5, min(5.0, sp))
        except Exception:
            pass

        # slightly slowed / smoothed
        wait = max(0.9, min(2.5, wait + 0.4))

        elapsed = 0.0
        step = 0.1
        while elapsed < wait:
            if stop_event.is_set():
                break
            time.sleep(step)
            elapsed += step

    # finished or stopped
    try:
        stream_q.put({"__done__": True}, timeout=1)
    except Exception:
        pass

def event_stream():
    while True:
        if stop_event.is_set():
            yield f"event: done\ndata: {json.dumps({'status': 'stopped'})}\n\n"
            break

        try:
            item = stream_q.get(timeout=1.0)
        except Empty:
            continue

        if isinstance(item, dict) and item.get("__done__"):
            yield f"event: done\ndata: {json.dumps({'status': 'done'})}\n\n"
            break

        payload = json.dumps(item)
        yield f"data: {payload}\n\n"

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_sim", methods=["POST"])
def start_sim():
    """
    Start simulation for a selected dataset from its CURRENT_INDEX.
    If we've reached the end, wrap to the beginning.
    """
    global sim_thread, CURRENT_INDEX, CURRENT_DATASET

    if sim_thread and sim_thread.is_alive():
        return jsonify({"status": "error", "message": "Simulation already running"}), 400

    data = request.get_json(silent=True) or {}
    dataset = data.get("dataset", "STEAD")
    if dataset not in DATASETS:
        return jsonify({"status": "error", "message": f"Unknown dataset: {dataset}"}), 400

    try:
        rows = get_manifest(dataset)
    except Exception as e:
        print("Error loading manifest:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

    if not rows:
        return jsonify({"status": "error", "message": f"Manifest CSV for {dataset} has no rows."}), 500

    # get current pointer (default 0)
    start_idx = CURRENT_INDEX.get(dataset, 0)
    if start_idx >= len(rows):
        start_idx = 0
        CURRENT_INDEX[dataset] = 0

    # clear queue from any previous run
    while not stream_q.empty():
        try:
            stream_q.get_nowait()
        except Exception:
            break

    stop_event.clear()
    CURRENT_DATASET = dataset
    sim_thread = threading.Thread(
        target=sim_loop, args=(dataset, start_idx), daemon=True
    )
    sim_thread.start()
    return jsonify({
        "status": "started",
        "dataset": dataset,
        "from_index": start_idx,
        "n_records": len(rows)
    }), 200

@app.route("/stop_sim", methods=["POST"])
def stop_sim():
    """
    Stop streaming, but keep CURRENT_INDEX[dataset] so that /start_sim
    will continue from the next event for that dataset.
    """
    global CURRENT_DATASET
    stop_event.set()
    try:
        stream_q.put({"__done__": True}, timeout=0.5)
    except Exception:
        pass

    next_index = None
    if CURRENT_DATASET is not None:
        next_index = CURRENT_INDEX.get(CURRENT_DATASET, 0)

    return jsonify({
        "status": "stopped",
        "dataset": CURRENT_DATASET,
        "next_index": next_index
    }), 200

@app.route("/stream")
def stream():
    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/metrics")
def metrics():
    """
    Returns OFFLINE metrics for the performance panel.

    Uses models/metrics_cross.json for per-dataset metrics.
    """
    # which dataset? default = STEAD
    dataset = request.args.get("dataset", "STEAD")

    # defaults (STEAD original if cross file not found)
    default_classification = {
        "accuracy": 0.997,
        "precision": 0.999,
        "recall": 0.997,
        "f1": 0.998
    }
    default_regression = {
        "mag_mae": 0.29,
        "dist_mae_km": 11.8,
        "sp_mae_s": 0.36
    }

    cross = METRICS_CROSS or {}

    # choose dataset key if present, else fall back to STEAD
    if dataset not in cross:
        dataset_key = "STEAD"
    else:
        dataset_key = dataset

    ds_metrics = cross.get(dataset_key, {}) if isinstance(cross, dict) else {}
    classification = ds_metrics.get("classification", default_classification)
    regression = ds_metrics.get("regression", default_regression)
    display_name = ds_metrics.get("display_name", dataset_key)

    return jsonify({
        "dataset": dataset_key,
        "display_name": display_name,
        "classification": classification,
        "regression": regression,
        "operational": {
            "inference_ms": MODELS_CFG.get("inference_ms", 45)
        }
    })

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting EEW demo server at http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True)


