# train_multitask.py
# Multitask training: classification (EQ/noise) + S-P (sec) + Distance (km) + Magnitude (Mw)

import os, time, math, json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---- Your paths -------------------------------------------------------------
CSV_BASE   = r"D:\datasets\stead_subset\subset.csv"       # original subset (for H5Dataset order)
CSV_MT     = r"D:\datasets\stead_subset\subset_mt.csv"    # labels with sp/dist/mag
H5         = r"D:\datasets\stead_subset\subset.hdf5"
OUT_DIR    = r"C:\Users\USER\eew\models\multitask_v1"
os.makedirs(OUT_DIR, exist_ok=True)
BEST_PATH  = os.path.join(OUT_DIR, "mt_v1.pt")
LOG_JSON   = os.path.join(OUT_DIR, "train_log.json")

# ---- Hyperparameters --------------------------------------------------------
EPOCHS = 12          # start small (6–12), increase later
BATCH  = 64
LR     = 2e-3
W_CLS  = 1.0         # loss weights for the four heads
W_SP   = 1.0
W_DST  = 1.0
W_MAG  = 1.0
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Data: use your streaming HDF5 Dataset (spec, label) -------------------
from dataloader_h5 import H5Dataset  # uses your wf_to_spec internally and returns (tensor, label)

base_ds = H5Dataset(CSV_BASE, H5)             # (spec: torch.FloatTensor [3,128,128], label: tensor int64)
mt_df   = pd.read_csv(CSV_MT)                  # same order/length as CSV_BASE
assert len(base_ds) == len(mt_df), "subset.csv and subset_mt.csv must have the same order/length."

# Prepare classification labels for stratified split
y_all = mt_df["label_eq"].astype(int).values
idx   = np.arange(len(mt_df))

i_tr, i_tmp, y_tr, y_tmp = train_test_split(idx, y_all, test_size=0.20, random_state=42, stratify=y_all)
i_va, i_te, _, _         = train_test_split(i_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

def collate(index_batch):
    """
    Build a batch for multitask:
      xs   : (B,3,128,128) float32
      ycls : (B,) int64
      ysp  : (B,) float32 (seconds)
      ydst : (B,) float32 (log1p(km))
      ymag : (B,) float32 (Mw)
      msp/mdst/mmag : (B,) float32 masks (1 if present else 0)
    """
    xs, ycls = [], []
    ysp, ydst, ymag = [], [], []
    msp, mdst, mmag = [], [], []

    for i in index_batch:
        # spec, label from H5Dataset (spec may already be Tensor)
        spec, lbl = base_ds[int(i)]
        if not isinstance(spec, torch.Tensor):
            spec = torch.from_numpy(spec)
        xs.append(spec.float())

        # label tensor -> python int
        ycls.append(int(lbl.item()) if isinstance(lbl, torch.Tensor) else int(lbl))

        row = mt_df.iloc[int(i)]

        # S-P (sec)
        sp = row["sp_sec"]
        if pd.notna(sp) and np.isfinite(sp):
            ysp.append(float(sp)); msp.append(1.0)
        else:
            ysp.append(0.0);       msp.append(0.0)

        # Distance (km) -> train on log1p for stability
        dist = row["dist_km"]
        if pd.notna(dist) and np.isfinite(dist) and dist >= 0:
            ydst.append(float(np.log1p(dist))); mdst.append(1.0)
        else:
            ydst.append(0.0);                       mdst.append(0.0)

        # Magnitude
        mag = row["mag"]
        if pd.notna(mag) and np.isfinite(mag):
            ymag.append(float(mag)); mmag.append(1.0)
        else:
            ymag.append(0.0);        mmag.append(0.0)

    return (
        torch.stack(xs, dim=0),
        torch.tensor(ycls, dtype=torch.long),
        torch.tensor(ysp,  dtype=torch.float32),
        torch.tensor(ydst, dtype=torch.float32),
        torch.tensor(ymag, dtype=torch.float32),
        torch.tensor(msp,  dtype=torch.float32),
        torch.tensor(mdst, dtype=torch.float32),
        torch.tensor(mmag, dtype=torch.float32),
    )

class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, ids): self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, k): return self.ids[k]

train_ld = DataLoader(IndexDataset(i_tr), batch_size=BATCH, shuffle=True,
                      num_workers=NUM_WORKERS, collate_fn=collate,
                      pin_memory=torch.cuda.is_available())
val_ld   = DataLoader(IndexDataset(i_va), batch_size=BATCH, shuffle=False,
                      num_workers=NUM_WORKERS, collate_fn=collate,
                      pin_memory=torch.cuda.is_available())
test_ld  = DataLoader(IndexDataset(i_te), batch_size=BATCH, shuffle=False,
                      num_workers=NUM_WORKERS, collate_fn=collate,
                      pin_memory=torch.cuda.is_available())

print(f"Split sizes: train={len(i_tr)}  val={len(i_va)}  test={len(i_te)}")

# ---- Model -----------------------------------------------------------------
class MTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),  # 64x64
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2), # 32x32
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # -> [B,64,1,1]
        )
        self.cls = nn.Linear(64, 2)
        self.sp  = nn.Linear(64, 1)
        self.dst = nn.Linear(64, 1)
        self.mag = nn.Linear(64, 1)

    def forward(self, x):
        z = self.backbone(x).view(x.size(0), -1)      # [B,64]
        return self.cls(z), self.sp(z), self.dst(z), self.mag(z)

model = MTNet().to(device)

# Class imbalance handling (optional)
cls_counts = np.bincount(y_tr, minlength=2)
w = cls_counts.sum() / (cls_counts + 1e-6)
w = (w / w.mean()).astype(np.float32)
cls_criterion = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))

mse = nn.MSELoss(reduction='none')

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

# ---- Train / Eval -----------------------------------------------------------
def run_epoch(loader, train=True):
    model.train(train)
    tot_loss = 0.0
    n = 0
    y_true, y_pred = [], []

    for xb, yb, sp, dst, mag, msp, mdst, mmag in tqdm(loader, leave=False):
        xb   = xb.to(device, non_blocking=True)
        yb   = yb.to(device, non_blocking=True)
        sp   = sp.to(device, non_blocking=True)
        dst  = dst.to(device, non_blocking=True)
        mag  = mag.to(device, non_blocking=True)
        msp  = msp.to(device, non_blocking=True)
        mdst = mdst.to(device, non_blocking=True)
        mmag = mmag.to(device, non_blocking=True)

        if train: optimizer.zero_grad(set_to_none=True)

        logits, sp_hat, dst_hat, mag_hat = model(xb)

        # Losses
        loss_cls = cls_criterion(logits, yb)

        loss_sp  = (mse(sp_hat.squeeze(1),  sp)  * msp).sum()  / (msp.sum()  + 1e-6)
        loss_dst = (mse(dst_hat.squeeze(1), dst) * mdst).sum() / (mdst.sum() + 1e-6)
        loss_mag = (mse(mag_hat.squeeze(1), mag) * mmag).sum() / (mmag.sum() + 1e-6)

        loss = W_CLS*loss_cls + W_SP*loss_sp + W_DST*loss_dst + W_MAG*loss_mag

        if train:
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        tot_loss += loss.item() * bs
        n += bs

        preds = logits.argmax(1)
        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc    = accuracy_score(y_true, y_pred)
    p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    return tot_loss/n, acc, p, r, f1

history = {"epoch": [], "tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": [], "va_f1": []}
best_va = -1.0

for ep in range(1, EPOCHS+1):
    t0 = time.time()
    tr_loss, tr_acc, _, _, _ = run_epoch(train_ld, train=True)
    va_loss, va_acc, _, _, va_f1 = run_epoch(val_ld,   train=False)
    scheduler.step(va_loss)

    dt = time.time() - t0
    print(f"Epoch {ep:02d}/{EPOCHS} | "
          f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
          f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f} | {dt:.1f}s")

    history["epoch"].append(ep)
    history["tr_loss"].append(tr_loss); history["tr_acc"].append(tr_acc)
    history["va_loss"].append(va_loss); history["va_acc"].append(va_acc); history["va_f1"].append(va_f1)

    if va_acc > best_va:
        best_va = va_acc
        torch.save(model.state_dict(), BEST_PATH)
        print("  -> saved best:", BEST_PATH)

# Save training log
with open(LOG_JSON, "w") as f:
    json.dump(history, f, indent=2)

# ---- Test with best checkpoint --------------------------------------------
model.load_state_dict(torch.load(BEST_PATH, map_location=device))
model.eval()
te_loss, te_acc, te_p, te_r, te_f1 = run_epoch(test_ld, train=False)
print("\nTEST  acc={:.3f}  P={:.3f}  R={:.3f}  F1={:.3f}".format(te_acc, te_p, te_r, te_f1))

print("\nSaved:")
print(" - best model :", BEST_PATH)
print(" - log json   :", LOG_JSON)
