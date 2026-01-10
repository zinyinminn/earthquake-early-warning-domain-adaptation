# train_usgs_regression.py
#
# Train a USGS regression model:
#   - Magnitude (Mw)
#   - Distance (km)
#   - S–P time (seconds)
#
# Inputs:
#   CSV: C:\Users\USER\eew_demo\data\usgs_big_reg\usgs_regression.csv
#   H5 : C:\Users\USER\eew_demo\data\usgs_reg.hdf5
#
# Outputs:
#   Model  : C:\Users\USER\eew\models\usgs_reg\usgs_regression_mt.pt
#   Metrics: C:\Users\USER\eew\models\usgs_reg\usgs_regression_metrics.json

import os
import json
import math
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ----------------- Paths -----------------
BASE_DEMO = r"C:\Users\USER\eew_demo"
DATA_DIR  = os.path.join(BASE_DEMO, "data")
CSV_PATH  = os.path.join(DATA_DIR, "usgs_big_reg", "usgs_regression.csv")
H5_PATH   = os.path.join(DATA_DIR, "usgs_reg.hdf5")

OUT_DIR   = r"C:\Users\USER\eew\models\usgs_reg"
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(OUT_DIR, "usgs_regression_mt.pt")
METRICS_JSON = os.path.join(OUT_DIR, "usgs_regression_metrics.json")

# ----------------- Hyperparameters -----------------
EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_MAG  = 1.0
W_DIST = 1.0
W_SP   = 1.0

# ----------------- Utils -----------------
def pick_column(row, candidates):
    """Return first finite numeric value among candidate column names, else None."""
    for c in candidates:
        if c in row.index:
            val = row[c]
            try:
                v = float(val)
                if math.isfinite(v):
                    return v
            except Exception:
                continue
    return None


# ----------------- Dataset -----------------
class USGSRegDataset(Dataset):
    """
    Loads spectrograms from HDF5 and regression targets from CSV.

    Targets:
      - mag_true   (float)
      - dist_log   (float)  = log1p(distance_km)
      - sp_true    (float)  (seconds)

    For missing values, target=0 and mask=0, so that loss ignores them.
    """

    def __init__(self, csv_path, h5_path, indices):
        super().__init__()
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.indices = np.array(indices, dtype=int)

        # open h5 lazily; we re-open in __getitem__ to avoid Windows HDF5 locking issues
        self.h5_path = h5_path

        # Cache column existence
        cols = set(self.df.columns)
        self.has_mag  = any(c in cols for c in ["mag", "magnitude", "Mw"])
        self.has_dist = any(c in cols for c in ["dist_km", "distance_km", "dist"])
        self.has_sp   = any(c in cols for c in ["sp_sec", "sp", "sp_time"])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        row = self.df.iloc[i]
        trace_name = str(row["trace_name"])

        # load spec from H5
        with h5py.File(self.h5_path, "r") as h5:
            spec = np.array(h5[trace_name], dtype=np.float32)  # (3, H, W)

        x = torch.from_numpy(spec).float()

        # ---- Magnitude ----
        mag_val = pick_column(row, ["mag", "magnitude", "Mw"])
        if mag_val is None:
            y_mag = 0.0
            m_mag = 0.0
        else:
            y_mag = float(mag_val)
            m_mag = 1.0

        # ---- Distance (km -> log1p) ----
        dist_val = pick_column(row, ["dist_km", "distance_km", "dist"])
        if dist_val is None or dist_val < 0:
            y_dist = 0.0
            m_dist = 0.0
        else:
            y_dist = float(np.log1p(dist_val))
            m_dist = 1.0

        # ---- S–P time (seconds) ----
        sp_val = pick_column(row, ["sp_sec", "sp", "sp_time"])
        if sp_val is None or sp_val < 0:
            y_sp = 0.0
            m_sp = 0.0
        else:
            y_sp = float(sp_val)
            m_sp = 1.0

        return (
            x,
            torch.tensor(y_mag,  dtype=torch.float32),
            torch.tensor(y_dist, dtype=torch.float32),
            torch.tensor(y_sp,   dtype=torch.float32),
            torch.tensor(m_mag,  dtype=torch.float32),
            torch.tensor(m_dist, dtype=torch.float32),
            torch.tensor(m_sp,   dtype=torch.float32),
        )


# ----------------- Model -----------------
class USGSRegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 64x64

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 32x32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B,64,1,1]
        )
        self.mag_head  = nn.Linear(64, 1)
        self.dist_head = nn.Linear(64, 1)
        self.sp_head   = nn.Linear(64, 1)

    def forward(self, x):
        z = self.backbone(x).view(x.size(0), -1)  # [B, 64]
        mag  = self.mag_head(z).squeeze(1)
        dist = self.dist_head(z).squeeze(1)
        sp   = self.sp_head(z).squeeze(1)
        return mag, dist, sp


# ----------------- Training helpers -----------------
def run_epoch(loader, model, optimizer=None):
    train = optimizer is not None
    if train:
        model.train()
    else:
        model.eval()

    mse = nn.MSELoss(reduction="none")

    total_loss = 0.0
    n_samples = 0

    # for metrics
    all_mag_true, all_mag_pred = [], []
    all_dist_true, all_dist_pred = [], []
    all_sp_true, all_sp_pred = [], []

    for batch in tqdm(loader, leave=False):
        xb, y_mag, y_dist, y_sp, m_mag, m_dist, m_sp = batch
        xb     = xb.to(DEVICE, non_blocking=True)
        y_mag  = y_mag.to(DEVICE)
        y_dist = y_dist.to(DEVICE)
        y_sp   = y_sp.to(DEVICE)
        m_mag  = m_mag.to(DEVICE)
        m_dist = m_dist.to(DEVICE)
        m_sp   = m_sp.to(DEVICE)

        if train:
            optimizer.zero_grad(set_to_none=True)

        mag_hat, dist_hat, sp_hat = model(xb)

        # losses with masks
        loss_mag  = (mse(mag_hat,  y_mag)  * m_mag).sum()  / (m_mag.sum()  + 1e-6)
        loss_dist = (mse(dist_hat, y_dist) * m_dist).sum() / (m_dist.sum() + 1e-6)
        loss_sp   = (mse(sp_hat,   y_sp)   * m_sp).sum()   / (m_sp.sum()   + 1e-6)

        loss = W_MAG * loss_mag + W_DIST * loss_dist + W_SP * loss_sp

        if train:
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        # collect for metrics (only where mask=1)
        with torch.no_grad():
            # magnitude
            mask = m_mag > 0.5
            if mask.any():
                all_mag_true.append(y_mag[mask].cpu().numpy())
                all_mag_pred.append(mag_hat[mask].cpu().numpy())

            # distance (convert back from log1p)
            mask = m_dist > 0.5
            if mask.any():
                true_d = torch.expm1(y_dist[mask])
                pred_d = torch.expm1(dist_hat[mask])
                all_dist_true.append(true_d.cpu().numpy())
                all_dist_pred.append(pred_d.cpu().numpy())

            # S–P
            mask = m_sp > 0.5
            if mask.any():
                all_sp_true.append(y_sp[mask].cpu().numpy())
                all_sp_pred.append(sp_hat[mask].cpu().numpy())

    avg_loss = total_loss / max(1, n_samples)

    def compute_mae_rmse(true_list, pred_list):
        if not true_list:
            return None, None
        t = np.concatenate(true_list)
        p = np.concatenate(pred_list)
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        return mae, rmse

    mag_mae,  mag_rmse  = compute_mae_rmse(all_mag_true,  all_mag_pred)
    dist_mae, dist_rmse = compute_mae_rmse(all_dist_true, all_dist_pred)
    sp_mae,   sp_rmse   = compute_mae_rmse(all_sp_true,   all_sp_pred)

    metrics = {
        "loss": avg_loss,
        "mag_mae": mag_mae,
        "mag_rmse": mag_rmse,
        "dist_mae_km": dist_mae,
        "dist_rmse_km": dist_rmse,
        "sp_mae_s": sp_mae,
        "sp_rmse_s": sp_rmse,
    }
    return metrics


# ----------------- Main -----------------
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not os.path.exists(H5_PATH):
        raise FileNotFoundError(f"HDF5 not found: {H5_PATH}")

    print("[INFO] Loading CSV:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    print("[INFO] Rows in CSV:", len(df))

    indices = np.arange(len(df))

    # simple random split: 70% train, 15% val, 15% test
    i_train, i_tmp = train_test_split(indices, test_size=0.30, random_state=42)
    i_val, i_test  = train_test_split(i_tmp,    test_size=0.50, random_state=42)

    print(f"[INFO] Split sizes: train={len(i_train)}  val={len(i_val)}  test={len(i_test)}")

    train_ds = USGSRegDataset(CSV_PATH, H5_PATH, i_train)
    val_ds   = USGSRegDataset(CSV_PATH, H5_PATH, i_val)
    test_ds  = USGSRegDataset(CSV_PATH, H5_PATH, i_test)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = USGSRegNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_metrics = None

    print("[INFO] Device:", DEVICE)

    for ep in range(1, EPOCHS + 1):
        print(f"\n[Epoch {ep}] -----------------------")

        tr_metrics = run_epoch(train_ld, model, optimizer)
        va_metrics = run_epoch(val_ld, model, optimizer=None)

        print(
            f"Train: loss={tr_metrics['loss']:.4f} | "
            f"mag_MAE={tr_metrics['mag_mae']:.3f} dist_MAE={tr_metrics['dist_mae_km']:.2f}km sp_MAE={tr_metrics['sp_mae_s']:.3f}s"
        )
        print(
            f"Valid: loss={va_metrics['loss']:.4f} | "
            f"mag_MAE={va_metrics['mag_mae']:.3f} dist_MAE={va_metrics['dist_mae_km']:.2f}km sp_MAE={va_metrics['sp_mae_s']:.3f}s"
        )

        if va_metrics["loss"] < best_val:
            best_val = va_metrics["loss"]
            best_metrics = va_metrics
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"[INFO] Saved new BEST model to: {CKPT_PATH}")

    print("\n[INFO] Training done.")
    print("[INFO] Loading best model and evaluating on TEST set...")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

    te_metrics = run_epoch(test_ld, model, optimizer=None)

    print(
        f"[TEST] loss={te_metrics['loss']:.4f} | "
        f"mag_MAE={te_metrics['mag_mae']:.3f} (RMSE={te_metrics['mag_rmse']:.3f}) | "
        f"dist_MAE={te_metrics['dist_mae_km']:.2f}km (RMSE={te_metrics['dist_rmse_km']:.2f}) | "
        f"sp_MAE={te_metrics['sp_mae_s']:.3f}s (RMSE={te_metrics['sp_rmse_s']:.3f})"
    )

    out_metrics = {
        "val": best_metrics,
        "test": te_metrics,
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(out_metrics, f, indent=2)
    print("[INFO] Saved metrics JSON to:", METRICS_JSON)


if __name__ == "__main__":
    main()
