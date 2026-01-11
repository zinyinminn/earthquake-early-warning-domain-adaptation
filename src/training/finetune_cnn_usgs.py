"""
finetune_cnn_usgs.py  (v2)

Stronger fine-tuning of the CNN baseline classifier (EQ vs Noise) on the USGS dataset.

- Loads the original STEAD-trained cnn_baseline.pt
- Fine-tunes on usgs_real.csv / usgs_real.hdf5
- Uses class-balancing (WeightedRandomSampler)
- Trains longer with smaller LR
- Saves cnn_baseline_usgs_ft.pt

Usage (from eew_demo folder):

    python finetune_cnn_usgs.py
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.signal import spectrogram, butter, filtfilt

# ================== PATHS  ==================
ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")

USGS_CSV = os.path.join(DATA_DIR, "usgs_real.csv")
USGS_H5  = os.path.join(DATA_DIR, "usgs_real.hdf5")

# Original STEAD-trained classifier
CKPT_CLS_INIT = r"C:\Users\USER\eew\models\cnn_baseline\cnn_baseline.pt"

# Where to save fine-tuned classifier
CKPT_CLS_FT   = r"C:\Users\USER\eew\models\cnn_baseline\cnn_baseline_usgs_ft.pt"

LOG_JSON      = os.path.join(os.path.dirname(CKPT_CLS_FT), "finetune_usgs_log.json")

# ================== HYPERPARAMETERS ==================
EPOCHS      = 40          # more epochs
BATCH_SIZE  = 64
LR          = 5e-5        # smaller LR
NUM_WORKERS = 0           # keep 0 because of HDF5
VAL_SIZE    = 0.2
FS          = 100.0
SPEC_SIZE   = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== MODEL (same as eval_cross_dataset.py) ==================
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

# ================== UTILS ==================
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
    padlen = 3*max(len(a), len(b))
    if data.shape[-1] <= padlen:
        return data
    return filtfilt(b, a, data)

def wf_to_spec(wf, fs=100, nperseg=128, noverlap=64, size=128, z_norm=True):
    x = np.array(wf, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[0] > x.shape[1] and x.shape[1] <= 16:
        x = x.T
    if x.shape[0] not in (1,2,3) and x.shape[1] in (1,2,3):
        x = x.T
    if x.shape[0] < 3:
        reps = int(np.ceil(3/x.shape[0]))
        x = np.tile(x, (reps,1))[:3]
    elif x.shape[0] > 3:
        x = x[:3]
    x = x - x.mean(axis=1, keepdims=True)

    chans = []
    for ch in range(3):
        nps = min(nperseg, x.shape[1])
        nov = min(noverlap, nps-1) if nps>1 else 0
        f, t, Sxx = spectrogram(
            x[ch], fs=fs, nperseg=nps, noverlap=nov,
            scaling="spectrum", mode="magnitude", detrend=False
        )
        Sxx = np.log1p(Sxx).astype(np.float32)
        out = np.zeros((size, size), dtype=np.float32)
        h = min(size, Sxx.shape[0]); w = min(size, Sxx.shape[1])
        out[:h, :w] = Sxx[:h, :w]
        chans.append(out)
    spec = np.stack(chans, axis=0)
    if z_norm:
        m = spec.mean(); s = spec.std() + 1e-6
        spec = (spec - m) / s
    return spec.astype(np.float32)

# ================== DATASET ==================
class USGSSpecDataset(Dataset):
    def __init__(self, df, h5_path, f_lo=1.0, f_hi=20.0):
        self.df = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.f_lo = f_lo
        self.f_hi = f_hi

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 not found: {h5_path}")
        self.h5 = h5py.File(h5_path, "r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tn = str(row["trace_name"])
        label = int(row["label_eq"])

        if tn not in self.h5:
            wf = np.zeros((3, 400), dtype=np.float32)
        else:
            wf = np.array(self.h5[tn])

        for c in range(min(3, wf.shape[0])):
            wf[c] = bandpass_filter(wf[c], FS, self.f_lo, self.f_hi)
        spec = wf_to_spec(wf, fs=FS, nperseg=128, noverlap=64, size=SPEC_SIZE, z_norm=True)
        spec = torch.from_numpy(spec).float()
        label = torch.tensor(label, dtype=torch.long)
        return spec, label

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass

# ================== TRAINING LOOP ==================
def run_epoch(model, loader, criterion, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    n = 0
    y_true, y_pred = [], []

    for xb, yb in tqdm(loader, leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs

        preds = logits.argmax(1).detach().cpu().numpy()
        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(preds)

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    avg_loss = total_loss / n
    return avg_loss, acc, p, r, f1

def main():
    if not os.path.exists(USGS_CSV):
        raise FileNotFoundError(f"USGS CSV not found: {USGS_CSV}")
    if not os.path.exists(USGS_H5):
        raise FileNotFoundError(f"USGS HDF5 not found: {USGS_H5}")
    if not os.path.exists(CKPT_CLS_INIT):
        raise FileNotFoundError(f"Initial classifier checkpoint not found: {CKPT_CLS_INIT}")

    print("[INFO] Loading USGS CSV:", USGS_CSV)
    df = pd.read_csv(USGS_CSV)

    if "label_eq" not in df.columns or "trace_name" not in df.columns:
        raise ValueError("USGS CSV must have 'trace_name' and 'label_eq' columns.")

    df = df.dropna(subset=["label_eq", "trace_name"]).copy()
    df["label_eq"] = df["label_eq"].astype(int)

    print(f"[INFO] Total usable rows in USGS CSV: {len(df)}")

    idx = np.arange(len(df))
    y = df["label_eq"].values
    i_tr, i_va = train_test_split(
        idx, test_size=VAL_SIZE, random_state=42, stratify=y
    )
    df_tr = df.iloc[i_tr].reset_index(drop=True)
    df_va = df.iloc[i_va].reset_index(drop=True)

    print(f"[INFO] Split sizes: train={len(df_tr)}  val={len(df_va)}")

    ds_tr = USGSSpecDataset(df_tr, USGS_H5)
    ds_va = USGSSpecDataset(df_va, USGS_H5)

    # build model and load initial weights
    model = CNNCls().to(device)
    print("[INFO] Loading initial classifier weights:", CKPT_CLS_INIT)
    state = safe_load(CKPT_CLS_INIT, map_location=device)
    state = remap_prefix(state, [("net.", "bb.net.")])
    model.load_state_dict(state, strict=False)

    # class weights & sampler
    cls_counts = np.bincount(df_tr["label_eq"].values, minlength=2)
    print(f"[INFO] Class counts (train) = {cls_counts}")

    # WeightedRandomSampler to balance classes
    # weight for each sample = 1 / class_count[label]
    class_weights = 1.0 / (cls_counts + 1e-6)
    sample_weights = class_weights[df_tr["label_eq"].values]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True
    )

    # loss also uses class weights (slight extra bias)
    w = cls_counts.sum() / (cls_counts + 1e-6)
    w = (w / w.mean()).astype(np.float32)
    cls_weights = torch.tensor(w, dtype=torch.float32).to(device)
    print(f"[INFO] Loss class weights = {w}")

    criterion = nn.CrossEntropyLoss(weight=cls_weights)

    train_ld = DataLoader(
        ds_tr, batch_size=BATCH_SIZE,
        sampler=sampler, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda")
    )
    val_ld = DataLoader(
        ds_va, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda")
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {
        "epoch": [],
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [],   "val_acc": [],   "val_f1": []
    }
    best_val_f1 = -1.0

    for ep in range(1, EPOCHS + 1):
        print(f"\n[INFO] Epoch {ep}/{EPOCHS}")
        tr_loss, tr_acc, tr_p, tr_r, tr_f1 = run_epoch(model, train_ld, criterion, optimizer)
        va_loss, va_acc, va_p, va_r, va_f1 = run_epoch(model, val_ld, criterion, optimizer=None)
        scheduler.step(va_loss)

        print(f"  Train: loss={tr_loss:.4f} acc={tr_acc:.3f} f1={tr_f1:.3f}")
        print(f"  Val  : loss={va_loss:.4f} acc={va_acc:.3f} f1={va_f1:.3f}")

        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), CKPT_CLS_FT)
            print(f"  -> saved best fine-tuned classifier to {CKPT_CLS_FT}")

    with open(LOG_JSON, "w") as f:
        json.dump(history, f, indent=2)
    print("\n[INFO] Training complete.")
    print("[INFO] Best val F1:", best_val_f1)
    print("[INFO] Log saved to:", LOG_JSON)

    ds_tr.close()
    ds_va.close()

if __name__ == "__main__":
    main()

