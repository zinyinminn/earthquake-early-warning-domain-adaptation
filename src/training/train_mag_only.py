# train_mag_only.py  — Magnitude-only training (physics-informed)
# Uses frozen regressor (S-P, Distance) predictions as inputs to the Mag head.

import os, json, time, math
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# -------------------- PATHS ---------------------------------
CSV_BASE   = r"D:\datasets\stead_subset\subset.csv"            # same order as H5
CSV_MT_FIX = r"D:\datasets\stead_subset\subset_mt_fixed.csv"   # preferred labels
CSV_MT_FALLBACK = r"D:\datasets\stead_subset\subset_mt.csv"    # fallback labels
H5_PATH    = r"D:\datasets\stead_subset\subset.hdf5"

CKPT_REG   = r"C:\Users\USER\eew\models\reg_only_v1\reg_only.pt"     # frozen S–P & Dist
SCALERS    = r"C:\Users\USER\eew\models\reg_only_v1\scalers.json"    # SP_MU / SP_STD

OUT_DIR    = r"C:\Users\USER\eew\models\mag_only_v1"
os.makedirs(OUT_DIR, exist_ok=True)
BEST_PT    = os.path.join(OUT_DIR, "mag_only.pt")
CALIB_JSON = os.path.join(OUT_DIR, "mag_calibration.json")
TRAIN_LOG  = os.path.join(OUT_DIR, "train_log.json")

BATCH = 64
EPOCHS = 15
LR = 2e-3
NUM_WORKERS = 0
HUBER_DELTA = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# -------------------- Dataset -------------------------------------------------
from dataloader_h5 import H5Dataset   # (spec[3,128,128], label 0/1)
ds_base = H5Dataset(CSV_BASE, H5_PATH)

# Prefer *_fixed, fallback to mt.csv
csv_path = CSV_MT_FIX if os.path.exists(CSV_MT_FIX) else CSV_MT_FALLBACK
mt = pd.read_csv(csv_path)
assert len(mt) == len(ds_base), "CSV and HDF5 dataset length mismatch."

# Normalize headers: some files use (label, sp, dist, mag)
rename_map = {
    "label": "label_eq",
    "sp": "sp_sec",
    "dist": "dist_km",
}
mt = mt.rename(columns={k: v for k, v in rename_map.items() if k in mt.columns})

# Coerce to numeric where relevant
for col in ["label_eq", "sp_sec", "dist_km", "mag"]:
    if col in mt.columns:
        mt[col] = pd.to_numeric(mt[col], errors="coerce")

# Build eq-mask:
# 1) If label_eq exists -> use it
# 2) Else assume the file is EQ-only (common for *_fixed)
if "label_eq" in mt.columns:
    mask_eq = (mt["label_eq"].fillna(1).astype(int) == 1)
else:
    mask_eq = np.ones(len(mt), dtype=bool)


mask_mag = np.isfinite(mt["mag"].astype(float)) if "mag" in mt.columns else np.zeros(len(mt), dtype=bool)
idx_all = np.where(mask_eq & mask_mag)[0]

# If still empty, fail early with a helpful message
if len(idx_all) == 0:
    raise RuntimeError(
        "No EQ rows with finite magnitude found. Check your CSV headers and values.\n"
        f"Columns present: {list(mt.columns)}\n"
        "Expected at least: trace_name, (label or label_eq), mag"
    )

i_tr, i_te = train_test_split(idx_all, test_size=0.20, random_state=42)
i_tr, i_va = train_test_split(i_tr,   test_size=0.20, random_state=42)

# For weighting (handle NaNs robustly)
mag_train = mt.loc[i_tr, "mag"].astype(float).values
bin_edges = np.array([-0.5, 1, 2, 3, 4, 10])
bin_ids   = np.digitize(mag_train, bin_edges)
counts    = np.bincount(bin_ids, minlength=len(bin_edges)+1).astype(float)
counts[counts==0] = 1.0
bin_w = 1.0 / counts
def mag_weights(values):
    b = np.digitize(values, bin_edges)
    w = bin_w[b]
    return (w / np.mean(w)).astype(np.float32)

# -------------------- Models --------------------------------------------------
class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
    def forward(self, x): return self.net(x).view(x.size(0), -1)

class RegressorFrozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.sp  = nn.Linear(64,1)
        self.dst = nn.Linear(64,1)
    def forward(self, x):
        z = self.backbone(x)
        return self.sp(z).squeeze(1), self.dst(z).squeeze(1)

class MagHeadPI(nn.Module):
    """Physics-informed Mag head: f([z, sp_sec_hat, log1p(dist_hat)]) → Mw"""
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.fc = nn.Sequential(
            nn.Linear(64 + 2, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x, sp_sec_hat, dist_km_hat):
        z = self.backbone(x)
        xpi = torch.cat([z, sp_sec_hat.unsqueeze(1), torch.log1p(dist_km_hat).unsqueeze(1)], dim=1)
        return self.fc(xpi).squeeze(1)

# Load frozen regressor
REG = RegressorFrozen().to(device).eval()
state = torch.load(CKPT_REG, map_location=device)
REG.load_state_dict(state)

with open(SCALERS, "r") as f:
    sc = json.load(f)
SP_MU, SP_STD = float(sc["SP_MU"]), float(sc["SP_STD"])

# Model to train
MAG = MagHeadPI().to(device)

# -------------------- Collate (compute reg preds on the fly) -----------------
def collate_mag(ids):
    xs, mag_t, wts, sp_hat, dk_hat = [], [], [], [], []
    for i in ids:
        spec, _ = ds_base[int(i)]
        if not isinstance(spec, torch.Tensor):
            spec = torch.from_numpy(spec)
        xs.append(spec.float())
        m = float(mt.at[int(i), "mag"])
        mag_t.append(m)
        wts.append(m)  # temporary store mag to compute weights
    X = torch.stack(xs, 0)

    with torch.no_grad():
        Xd = X.to(device)
        sp_z, dst_log = REG(Xd)
        sp_sec = sp_z * SP_STD + SP_MU
        dist_km = torch.expm1(dst_log)
        sp_hat = sp_sec.detach().cpu()
        dk_hat = dist_km.detach().cpu()

    w = mag_weights(np.array(wts, float))
    return X, torch.tensor(mag_t, dtype=torch.float32), torch.tensor(w, dtype=torch.float32), sp_hat, dk_hat

class Ids(torch.utils.data.Dataset):
    def __init__(self, arr): self.arr = list(arr)
    def __len__(self): return len(self.arr)
    def __getitem__(self, k): return self.arr[k]

train_ld = DataLoader(Ids(i_tr), batch_size=BATCH, shuffle=True,  num_workers=NUM_WORKERS, collate_fn=collate_mag, pin_memory=torch.cuda.is_available())
val_ld   = DataLoader(Ids(i_va), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_mag, pin_memory=torch.cuda.is_available())
test_ld  = DataLoader(Ids(i_te), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_mag, pin_memory=torch.cuda.is_available())

# -------------------- Train ---------------------------------------------------
huber = nn.HuberLoss(delta=HUBER_DELTA, reduction='none')
opt   = torch.optim.Adam(MAG.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, verbose=True)

def run_epoch(dl, train=True):
    MAG.train(train)
    n=0; sse=0.0; mae_sum=0.0; loss_sum=0.0
    all_t, all_h = [], []
    for X, mag_true, weights, sp_hat, dk_hat in tqdm(dl, leave=False):
        X = X.to(device, non_blocking=True)
        mag_true = mag_true.to(device)
        weights = weights.to(device)
        sp_hat = sp_hat.to(device)
        dk_hat = dk_hat.to(device)

        if train: opt.zero_grad(set_to_none=True)
        mag_pred = MAG(X, sp_hat, dk_hat)
        loss_raw = huber(mag_pred, mag_true)           
        loss = (loss_raw * weights).mean()
        if train:
            loss.backward()
            opt.step()

        bs = X.size(0)
        n += bs
        loss_sum += loss.item()*bs
        sse += torch.sum((mag_pred - mag_true)**2).item()
        mae_sum += torch.sum(torch.abs(mag_pred - mag_true)).item()
        all_t.append(mag_true.detach().cpu().numpy())
        all_h.append(mag_pred.detach().cpu().numpy())

    rmse = (sse/n)**0.5
    mae  = mae_sum/n
    return loss_sum/n, mae, rmse, np.concatenate(all_t), np.concatenate(all_h)

best_rmse = 1e9
log = {"epoch": [], "tr_mae": [], "va_mae": [], "va_rmse": []}

for ep in range(1, EPOCHS+1):
    tr_loss, tr_mae, _, _, _ = run_epoch(train_ld, True)
    va_loss, va_mae, va_rmse, va_t, va_h = run_epoch(val_ld, False)
    sched.step(va_loss)
    print(f"Epoch {ep:02d}: train MAE={tr_mae:.3f} | val MAE={va_mae:.3f} RMSE={va_rmse:.3f}")
    log["epoch"].append(ep); log["tr_mae"].append(tr_mae); log["va_mae"].append(va_mae); log["va_rmse"].append(va_rmse)

    if va_rmse < best_rmse:
        best_rmse = va_rmse
        torch.save(MAG.state_dict(), BEST_PT)
        # post-hoc linear calibration on validation set: mag_true ≈ a*mag_pred + b
        a, b = np.polyfit(va_h, va_t, deg=1)
        with open(CALIB_JSON, "w") as f:
            json.dump({"a": float(a), "b": float(b)}, f, indent=2)
        print(f"  -> saved best {BEST_PT} and calibration a={a:.3f}, b={b:.3f}")

with open(TRAIN_LOG, "w") as f:
    json.dump(log, f, indent=2)

# -------------------- Test with calibration ----------------------------------
MAG.load_state_dict(torch.load(BEST_PT, map_location=device))
MAG.eval()
_, te_mae, te_rmse, y_t, y_h = run_epoch(test_ld, False)
with open(CALIB_JSON, "r") as f:
    calib = json.load(f)
y_h_cal = calib["a"] * y_h + calib["b"]
mae_cal = float(np.mean(np.abs(y_h_cal - y_t)))
rmse_cal = float(np.sqrt(np.mean((y_h_cal - y_t)**2)))
print(f"\nTEST (raw)  MAE={te_mae:.3f} RMSE={te_rmse:.3f}")
print(f"TEST (cal.) MAE={mae_cal:.3f} RMSE={rmse_cal:.3f}")
print("Saved:\n -", BEST_PT, "\n -", CALIB_JSON, "\n -", TRAIN_LOG)

