
import os, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm


CSV = r"D:\datasets\stead_subset\subset_mt.csv"         # works with sp/sp_sec, dist/dist_km, label/label_eq...
H5  = r"D:\datasets\stead_subset\subset.hdf5"
MODEL_PATH = r"C:\Users\USER\eew\models\multitask_v1\mt_v1.pt"
OUT_DIR = os.path.join(os.path.dirname(MODEL_PATH), "eval"); os.makedirs(OUT_DIR, exist_ok=True)

DIST_WAS_LOG1P = True

# --------- PREPROCESS (match training) -----
FS=100; NPERSEG=128; NOVERLAP=64; SPEC_SIZE=128; DETREND='constant'; SCALING='spectrum'; Z_NORM=True
def _stft_mag(x):
    import scipy.signal as sg
    f,t,Z = sg.stft(x, fs=FS, window='hann', nperseg=NPERSEG,
                    noverlap=NOVERLAP, detrend=DETREND, boundary=None)
    S = np.log1p(np.abs(Z))
    F,T = S.shape
    if F < SPEC_SIZE or T < SPEC_SIZE:
        S = np.pad(S, ((0,max(0,SPEC_SIZE-F)), (0,max(0,SPEC_SIZE-T))), mode='edge')
    return S[:SPEC_SIZE,:SPEC_SIZE]

def wf_to_spec(wf_6000x3):
    ch=[]
    for k in range(3):
        S = _stft_mag(wf_6000x3[:,k])
        if Z_NORM:
            m,s = S.mean(), S.std()+1e-6
            S = (S-m)/s
        ch.append(S)
    return np.stack(ch,0).astype(np.float32)  # (3,128,128)

# --------- COLUMN RESOLUTION ----------------
def resolve_column(columns, candidates):
    for c in candidates:
        if c in columns: return c
    return None

def choose_label_column(columns):
    return resolve_column(columns, ["trace_category","label","label_eq","y","is_eq","is_earthquake"])

# --------- LABEL DETECTION -------------------
def detect_label_from_row(row, label_col):
    # Prefer explicit column
    if label_col is not None and label_col in row:
        val = row[label_col]
        try:
            if pd.notna(val) and float(val) in (0.0, 1.0) or isinstance(val, (int, np.integer)):
                return int(float(val) >= 0.5)
        except Exception:
            pass
        sv = str(val).strip().lower()
        if sv in ("1","true","eq","earthquake","quake"): return 1
        if sv in ("0","false","noise","no"): return 0
    # Fallback: derive from trace_name suffix
    name = str(row.get("trace_name",""))
    if name.endswith("_EV"): return 1
    if name.endswith("_NO"): return 0
    # Last fallback: assume noise
    return 0

# --------- DATASET (filters missing keys) ----
class H5Dataset_MT(Dataset):
    def __init__(self, csv_file, h5_file):
        raw = pd.read_csv(csv_file)
        # Resolve regression/label columns (flexible names)
        self.col_label = choose_label_column(raw.columns)
        self.col_sp    = resolve_column(raw.columns,  ["sp","sp_sec"])
        self.col_dist  = resolve_column(raw.columns,  ["dist","dist_km","source_distance_km"])
        self.col_mag   = resolve_column(raw.columns,  ["mag","magnitude","source_magnitude"])

        # Diagnostics
        print("Column binding:")
        print("  label ->", self.col_label or "(suffix _EV/_NO)")
        print("  sp    ->", self.col_sp or "(missing)")
        print("  dist  ->", self.col_dist or "(missing)")
        print("  mag   ->", self.col_mag or "(missing)")

        self.h5 = h5py.File(h5_file, "r")
        self.root = self.h5["data"] if "data" in self.h5 else self.h5
        h5_keys = set(self.root.keys())

        # Keep only rows present in HDF5
        exist_mask = raw["trace_name"].astype(str).isin(h5_keys)
        self.df = raw[exist_mask].reset_index(drop=True)
        print(f"Dataset filtered: {len(self.df)} rows kept, {len(raw)-len(self.df)} missing from HDF5 were dropped.")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key = str(row["trace_name"])
        wf = self.root[key][()]  # (6000,3) float32
        spec = wf_to_spec(wf)

        label = detect_label_from_row(row, self.col_label)

        # pull values with NaN -> masks
        def pull(col):
            if col is None: return np.nan
            v = row.get(col, np.nan)
            return v if pd.notna(v) else np.nan

        sp   = pull(self.col_sp)
        dist = pull(self.col_dist)
        mag  = pull(self.col_mag)

        sp_t   = 0.0 if np.isnan(sp)   else float(sp)
        dist_t = 0.0 if np.isnan(dist) else float(dist)
        mag_t  = 0.0 if np.isnan(mag)  else float(mag)
        msp    = 0.0 if np.isnan(sp)   else 1.0
        mdist  = 0.0 if np.isnan(dist) else 1.0
        mmag   = 0.0 if np.isnan(mag)  else 1.0

        return (
            torch.tensor(spec, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(sp_t, dtype=torch.float32),
            torch.tensor(dist_t, dtype=torch.float32),
            torch.tensor(mag_t, dtype=torch.float32),
            torch.tensor(msp, dtype=torch.float32),
            torch.tensor(mdist, dtype=torch.float32),
            torch.tensor(mmag, dtype=torch.float32),
        )

# --------- MODEL (match training) -----------
class MTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)))
        self.cls = nn.Linear(64,2)
        self.sp  = nn.Linear(64,1)
        self.dst = nn.Linear(64,1)
        self.mag = nn.Linear(64,1)
    def forward(self,x):
        z = self.backbone(x).view(x.size(0),-1)
        return self.cls(z), self.sp(z), self.dst(z), self.mag(z)

# --------- LOAD MODEL -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MTNet().to(device)

# Avoid security warning if supported
try:
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
except TypeError:
    state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# --------- DATALOADER -----------------------
ds = H5Dataset_MT(CSV, H5)
ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

# --------- EVAL -----------------------------
y_true, y_prob, y_pred = [], [], []
sp_true, sp_pred = [], []
dist_true, dist_pred = [], []
mag_true, mag_pred = [], []

with torch.no_grad():
    for xb,yb,sp,dist,mag,msp,mdist,mmag in tqdm(ld, total=len(ld), desc="Evaluating"):
        xb = xb.to(device)
        logits, sp_out, dist_out, mag_out = model(xb)

        probs = torch.softmax(logits,1)[:,1].cpu().numpy()
        pred  = logits.argmax(1).cpu().numpy()

        y_true += yb.numpy().tolist()
        y_prob += probs.tolist()
        y_pred += pred.tolist()

        # collect regression only where masks are 1 (ignore NaNs automatically)
        if torch.sum(msp) > 0:
            sp_true  += sp[msp>0].numpy().tolist()
            sp_pred  += sp_out[msp>0].cpu().numpy().squeeze().tolist()
        if torch.sum(mdist) > 0:
            dist_true += dist[mdist>0].numpy().tolist()
            d_pred = dist_out[mdist>0].cpu().numpy().squeeze()
            if DIST_WAS_LOG1P:
                d_pred = np.expm1(d_pred)
            dist_pred += d_pred.tolist()
        if torch.sum(mmag) > 0:
            mag_true += mag[mmag>0].numpy().tolist()
            mag_pred += mag_out[mmag>0].cpu().numpy().squeeze().tolist()

# ---- classification metrics
acc = accuracy_score(y_true, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
try:
    auc = roc_auc_score(y_true, y_prob)
except Exception:
    auc = float("nan")
rep_text = classification_report(y_true, y_pred, target_names=["noise","earthquake"], digits=3)
cm = confusion_matrix(y_true, y_pred).tolist()

# ---- regression metrics
def rmse(a,b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a)==0: return float("nan")
    return float(np.sqrt(np.mean((a-b)**2)))
def mae(a,b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a)==0: return float("nan")
    return float(np.mean(np.abs(a-b)))

metrics = {
    "classification": {
        "accuracy": float(acc), "precision": float(p), "recall": float(r),
        "f1": float(f1), "auc": float(auc),
        "confusion_matrix": cm, "report_text": rep_text
    },
    "regression": {
        "sp":   {"rmse": rmse(sp_true, sp_pred),     "mae": mae(sp_true, sp_pred),       "n": len(sp_true)},
        "dist": {"rmse": rmse(dist_true, dist_pred), "mae": mae(dist_true, dist_pred),   "n": len(dist_true)},
        "mag":  {"rmse": rmse(mag_true, mag_pred),   "mae": mae(mag_true, mag_pred),     "n": len(mag_true)}
    }
}

with open(os.path.join(OUT_DIR, "metrics_test.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(rep_text)
print(json.dumps(metrics["regression"], indent=2))
print("\nSaved metrics to:", os.path.join(OUT_DIR, "metrics_test.json"))

# ---- optional: residual histograms (nice for thesis)
def resid_hist(true, pred, title, path):
    if len(true)==0: return
    r = np.array(pred) - np.array(true)
    plt.figure(figsize=(5,3))
    plt.hist(r, bins=40)
    plt.title(title); plt.xlabel("residual"); plt.ylabel("count"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

resid_hist(sp_true,   sp_pred,   "S–P residuals (s)",       os.path.join(OUT_DIR,"resid_sp.png"))
resid_hist(dist_true, dist_pred, "Distance residuals (km)", os.path.join(OUT_DIR,"resid_dist.png"))
resid_hist(mag_true,  mag_pred,  "Magnitude residuals",     os.path.join(OUT_DIR,"resid_mag.png"))
print("Saved residual plots in:", OUT_DIR)

