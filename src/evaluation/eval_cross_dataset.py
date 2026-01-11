import os, argparse, json, math, warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from scipy.signal import spectrogram, butter, filtfilt
import matplotlib.pyplot as plt


CKPT_CLS = r"C:\Users\USER\eew\models\cnn_baseline\cnn_baseline.pt"
CKPT_REG = r"C:\Users\USER\eew\models\reg_only_v1\reg_only.pt"
SCALERS  = r"C:\Users\USER\eew\models\reg_only_v1\scalers.json"
CKPT_MAG = r"C:\Users\USER\eew\models\mag_only_v1\mag_only.pt"
MAG_CAL  = r"C:\Users\USER\eew\models\mag_only_v1\mag_calibration.json"

FS = 100.0
SPEC_SIZE = 128
THRESH_DEFAULT = 0.50

# -------------------- Models --------------------
class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
    def forward(self, x):
        return self.net(x).view(x.size(0), -1)

class CNNCls(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = TinyBackbone()
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        return self.fc(self.bb(x))

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = TinyBackbone()
        self.sp  = nn.Linear(64,1)
        self.dst = nn.Linear(64,1)
    def forward(self, x):
        z = self.bb(x)
        return self.sp(z).squeeze(1), self.dst(z).squeeze(1)

class MagHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = TinyBackbone()
        self.mag = nn.Linear(64,1)
    def forward(self, x):
        return self.mag(self.bb(x)).squeeze(1)

# -------------------- Utils --------------------
def safe_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def remap_prefix(state, mapping):
    from collections import OrderedDict
    new_state = OrderedDict()
    for k,v in state.items():
        replaced = False
        for old, new in mapping:
            if k.startswith(old):
                new_state[new + k[len(old):]] = v
                replaced = True
                break
        if not replaced:
            new_state[k] = v
    return new_state

def bandpass_filter(data, fs, f_lo=1.0, f_hi=20.0, order=4):
    nyq = 0.5 * fs
    low, high = max(1e-6, f_lo/nyq), min(0.999, f_hi/nyq)
    b,a = butter(order, [low, high], btype="band")
    # handle short traces gracefully
    padlen = 3*max(len(a), len(b))
    if data.shape[-1] <= padlen:
        return data
    return filtfilt(b, a, data)

def wf_to_spec(wf, fs=100, nperseg=128, noverlap=64, size=128, z_norm=True):
    """
    Same recipe as training:
      - De-mean per channel
      - spectrogram(scaling="spectrum", mode="magnitude")
      - log1p
      - top-left crop/pad to (size,size)
      - global z-norm
    Input wf: (T,3) or (3,T) or (T,) — we’ll coerce to (3,T) with repeats if needed.
    """
    x = np.array(wf, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]  # (1,T)
    # make (C,T)
    if x.shape[0] > x.shape[1] and x.shape[1] <= 16:
        x = x.T
    if x.shape[0] not in (1,2,3) and x.shape[1] in (1,2,3):
        x = x.T
    # ensure 3 channels
    if x.shape[0] < 3:
        reps = int(np.ceil(3/x.shape[0]))
        x = np.tile(x, (reps,1))[:3]
    elif x.shape[0] > 3:
        x = x[:3]
    # de-mean
    x = x - x.mean(axis=1, keepdims=True)

    chans = []
    for ch in range(3):
        # guard small signals
        nps = min(nperseg, x.shape[1])
        nov = min(noverlap, nps-1) if nps>1 else 0
        f, t, Sxx = spectrogram(
            x[ch], fs=fs, nperseg=nps, noverlap=nov,
            scaling="spectrum", mode="magnitude", detrend=False
        )
        Sxx = np.log1p(Sxx).astype(np.float32)
        # crop/pad to (size,size) top-left
        out = np.zeros((size, size), dtype=np.float32)
        h = min(size, Sxx.shape[0]); w = min(size, Sxx.shape[1])
        out[:h, :w] = Sxx[:h, :w]
        chans.append(out)
    spec = np.stack(chans, axis=0)
    if z_norm:
        m = spec.mean(); s = spec.std() + 1e-6
        spec = (spec - m) / s
    return spec.astype(np.float32)

def make_spec_tensor(batch_wf, fs, f_lo=None, f_hi=None, tta=False):
    """
    batch_wf: list of np.ndarray (C,T). We will convert to (B,3,128,128).
    If f_lo/f_hi provided, apply band-pass channel-wise before spec.
    If tta=True, we apply channel flips (cyclic permutations & sign flips) and average logits later outside.
    """
    specs = []
    for wf in batch_wf:
        x = np.array(wf, dtype=np.float32)
        # shape to (C,T)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[0] > x.shape[1] and x.shape[1] <= 16:
            x = x.T
        if x.shape[0] not in (1,2,3) and x.shape[1] in (1,2,3):
            x = x.T
        # ensure 3 channels
        if x.shape[0] < 3:
            reps = int(np.ceil(3/x.shape[0]))
            x = np.tile(x, (reps,1))[:3]
        elif x.shape[0] > 3:
            x = x[:3]
        # band-pass
        if (f_lo is not None) and (f_hi is not None):
            for c in range(3):
                x[c] = bandpass_filter(x[c], fs, f_lo, f_hi)
        # to spec
        spec = wf_to_spec(x, fs=fs, nperseg=128, noverlap=64, size=SPEC_SIZE, z_norm=True)
        specs.append(spec)
    X = torch.from_numpy(np.stack(specs)).float()
    return X

def softmax_temp(logits, T=1.0):
    if T is None or abs(T-1.0)<1e-6:
        return torch.softmax(logits, dim=1)
    return torch.softmax(logits / T, dim=1)

def search_best_threshold(y_true, y_prob):
    # maximize F1 over thresholds in [0.05..0.95]
    best = (0.5, 0.0)
    for th in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= th).astype(int)
        p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best[1]: best = (th, f1)
    return best

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV catalog (must include trace_name,label_eq)")
    ap.add_argument("--h5",  required=True, help="HDF5 with waveforms (datasets at root, each (C,T))")
    ap.add_argument("--limit", type=int, default=0, help="limit rows for quick test")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--amp", type=lambda s: s.lower()=="true", default=True)
    ap.add_argument("--tta", type=lambda s: s.lower()=="true", default=True)
    ap.add_argument("--temp", type=float, default=3.0, help="softmax temperature for classifier")
    ap.add_argument("--auto_threshold", choices=["f1", "none"], default="f1")
    ap.add_argument("--f_lo", type=float, default=1.0)
    ap.add_argument("--f_hi", type=float, default=20.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Starting eval_cross_dataset.py")
    print(f"[INFO] CSV={args.csv}")
    print(f"[INFO] H5 ={args.h5}")
    print(f"[INFO] spec=strict f_lo={args.f_lo} f_hi={args.f_hi} limit={args.limit} batch={args.batch} tta={args.tta} amp={args.amp}")

    df = pd.read_csv(args.csv)
    if args.limit>0:
        df = df.head(args.limit).copy()

    # dataset counts
    if "label_eq" not in df.columns or "trace_name" not in df.columns:
        raise ValueError("CSV must include trace_name, label_eq columns.")
    counts = df["label_eq"].value_counts(dropna=False)
    print("[INFO] label_eq counts:\n", counts.to_string())

    with h5py.File(args.h5, "r") as f:
        keys = set(f.keys())
    missing = [n for n in df["trace_name"].astype(str).tolist() if n not in keys]
    if len(missing) > 0:
        print(f"[WARN] Missing {len(missing)} trace_name in H5 (will skip those rows).")

    # ---------------- Load models ----------------
    # classifier
    CLS = CNNCls().to(device).eval()
    s = safe_load(CKPT_CLS, map_location=device)
    s = remap_prefix(s, [("net.", "bb.net.")])  # your classifier saved as TinyCNN().net.*
    CLS.load_state_dict(s, strict=False)

    # regression (S-P, Dist)
    REG = Regressor().to(device).eval()
    s = safe_load(CKPT_REG, map_location=device)
    s = remap_prefix(s, [("backbone.net.", "bb.net."), ("backbone.", "bb.")])
    REG.load_state_dict(s, strict=False)
    with open(SCALERS, "r") as fp:
        scalers = json.load(fp)
    SP_MU, SP_STD = float(scalers["SP_MU"]), float(scalers["SP_STD"])

    # magnitude head + calibration
    MAG = MagHead().to(device).eval()
    s = safe_load(CKPT_MAG, map_location=device)
    s = remap_prefix(s, [("backbone.net.", "bb.net."), ("backbone.", "bb.")])
    MAG.load_state_dict(s, strict=False)
    a, b = 1.0, 0.0
    if os.path.exists(MAG_CAL):
        cal = json.load(open(MAG_CAL, "r"))
        a, b = float(cal.get("a", 1.0)), float(cal.get("b", 0.0))
    print(f"Using magnitude calibration: a={a:.3f}, b={b:.3f}")

    # ---------------- Classifier eval (EQ + Noise) ----------------
    cls_rows = df[df["trace_name"].astype(str).isin(keys)]
    y_true, y_prob = [], []
    batch_wf, batch_lab = [], []

    with h5py.File(args.h5, "r") as f:
        for _, row in cls_rows.iterrows():
            tn = str(row["trace_name"])
            lab = int(row["label_eq"])
            x = np.array(f[tn])  # (C,T)
            batch_wf.append(x)
            batch_lab.append(lab)
            if len(batch_wf) == args.batch:
                X = make_spec_tensor(batch_wf, FS, args.f_lo, args.f_hi, tta=False).to(device)
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type=="cuda")):
                    logits = CLS(X)
                    probs  = softmax_temp(logits, T=args.temp)[:,1].detach().cpu().numpy()
                y_prob.extend(probs.tolist()); y_true.extend(batch_lab)
                batch_wf, batch_lab = [], []
        if batch_wf:
            X = make_spec_tensor(batch_wf, FS, args.f_lo, args.f_hi, tta=False).to(device)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type=="cuda")):
                logits = CLS(X)
                probs  = softmax_temp(logits, T=args.temp)[:,1].detach().cpu().numpy()
            y_prob.extend(probs.tolist()); y_true.extend(batch_lab)

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    if len(np.unique(y_true)) == 1:
        # one-class
        rate = float((y_prob>=THRESH_DEFAULT).mean())
        print(f"\nOne-class dataset (all {'earthquakes' if y_true[0]==1 else 'noise'}).")
        print(f"Detection Rate @ threshold {THRESH_DEFAULT:.2f}: {rate:.3f}  (predicted EQ on {(y_prob>=THRESH_DEFAULT).sum()}/{len(y_prob)})")
    else:
        # auto threshold by F1 or fixed default
        if args.auto_threshold == "f1":
            th, f1_cal = search_best_threshold(y_true, y_prob)
            print(f"\nClassifier on {os.path.basename(args.csv)}")
            print(f"  Temperature T={args.temp:.2f}  Adaptive threshold={th:.2f} (calib F1={f1_cal:.3f})")
            y_pred = (y_prob >= th).astype(int)
        else:
            th = THRESH_DEFAULT
            y_pred = (y_prob >= th).astype(int)
            print(f"\nClassifier on {os.path.basename(args.csv)} (fixed threshold={th:.2f})")

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        print(f"  Accuracy={acc:.3f}  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")

        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(4,4))
        ax = plt.gca()
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Noise","EQ"]); ax.set_yticklabels(["Noise","EQ"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        out_cm = os.path.join(os.path.dirname(args.csv), os.path.splitext(os.path.basename(args.csv))[0] + "_cm.png")
        plt.tight_layout(); plt.savefig(out_cm, dpi=160); plt.close(fig)
        print(f"Saved confusion matrix: {out_cm}")

    # ---------------- Regression (EQ-only, pred-only stats if no truth) ----------------
    eq_rows = df[(df["label_eq"].astype(int)==1) & (df["trace_name"].astype(str).isin(keys))]
    sp_pred, dst_pred, mag_pred, mag_true = [], [], [], []

    with h5py.File(args.h5, "r") as f:
        batch_wf = []
        idxs = []
        for idx, row in eq_rows.iterrows():
            tn = str(row["trace_name"])
            x = np.array(f[tn])  # (C,T)
            batch_wf.append(x); idxs.append(idx)
            if len(batch_wf) == args.batch:
                Xs = make_spec_tensor(batch_wf, FS, args.f_lo, args.f_hi, tta=False).to(device)
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type=="cuda")):
                    sp_z, dst_log = REG(Xs)
                    mag_raw = MAG(Xs)
                sp_pred.extend((sp_z.cpu().numpy().ravel()*SP_STD + SP_MU).tolist())
                dst_pred.extend(np.expm1(dst_log.cpu().numpy().ravel()).tolist())
                mag_pred.extend((a*mag_raw.cpu().numpy().ravel() + b).tolist())
                idxs = []; batch_wf = []
        if batch_wf:
            Xs = make_spec_tensor(batch_wf, FS, args.f_lo, args.f_hi, tta=False).to(device)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type=="cuda")):
                sp_z, dst_log = REG(Xs)
                mag_raw = MAG(Xs)
            sp_pred.extend((sp_z.cpu().numpy().ravel()*SP_STD + SP_MU).tolist())
            dst_pred.extend(np.expm1(dst_log.cpu().numpy().ravel()).tolist())
            mag_pred.extend((a*mag_raw.cpu().numpy().ravel() + b).tolist())

        # collect mag truth if present and finite
        if "mag" in eq_rows.columns:
            for _, row in eq_rows.iterrows():
                val = row["mag"]
                try:
                    v = float(val)
                    mag_true.append(v if math.isfinite(v) else np.nan)
                except Exception:
                    mag_true.append(np.nan)
        else:
            mag_true = [np.nan]*len(eq_rows)

    print("\nRegression (EQ-only)")
    if len(sp_pred) > 0 and np.isfinite(sp_pred).any():
        print(f"  S–P (pred only): mean={np.nanmean(sp_pred):.2f}s  (truth not provided)")
    else:
        print(f"  S–P (pred only): no predictions")
    if len(dst_pred) > 0 and np.isfinite(dst_pred).any():
        print(f"  Dist (pred only): mean={np.nanmean(dst_pred):.1f} km  (truth not provided)")
    else:
        print(f"  Dist (pred only): no predictions")

    mag_pred = np.array(mag_pred, dtype=float)
    mag_true = np.array(mag_true, dtype=float)
    mask_mag = np.isfinite(mag_true)
    if mask_mag.any():
        mae = float(np.mean(np.abs(mag_pred[mask_mag] - mag_true[mask_mag])))
        rmse = float(np.sqrt(np.mean((mag_pred[mask_mag] - mag_true[mask_mag])**2)))
        print(f"  Mag MAE={mae:.2f}  RMSE={rmse:.2f}  on n={int(mask_mag.sum())}")
        # residual plot
        fig = plt.figure(figsize=(5,3))
        err = mag_pred[mask_mag] - mag_true[mask_mag]
        plt.hist(err, bins=30)
        plt.xlabel("Pred-Truth (Mw)"); plt.ylabel("Count"); plt.title("Magnitude Residuals")
        out_png = os.path.join(os.path.dirname(args.csv), os.path.splitext(os.path.basename(args.csv))[0] + "_mag_resid.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close(fig)
        print(f"Saved magnitude residual histogram: {out_png}")
    else:
        print("  Mag: truth non-finite; skipped error metrics.")

    # Sample prints
    names_eq = eq_rows["trace_name"].astype(str).tolist()
    print("\nSample predictions (first 10 EQ rows):")
    for i, tn in enumerate(names_eq[:10]):
        sp = sp_pred[i] if i < len(sp_pred) else np.nan
        dk = dst_pred[i] if i < len(dst_pred) else np.nan
        mg = mag_pred[i] if i < len(mag_pred) else np.nan
        print(f"  {tn}: S-P≈{(np.nan if not np.isfinite(sp) else round(sp,2))}s  Dist≈{(np.nan if not np.isfinite(dk) else int(round(dk)))}km  Mag≈{(np.nan if not np.isfinite(mg) else round(mg,2))}")

    print("\n[INFO] Done.")

if __name__ == "__main__":
    main()

