# train_cnn_baseline.py  — CNN baseline with tqdm progress + early stopping (no verbose warning)
import os, time, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from dataloader_h5 import H5Dataset  

# -------------------- CONFIG --------------------
CSV = r"D:\datasets\stead_subset\subset.csv"
H5  = r"D:\datasets\stead_subset\subset.hdf5"
OUT = r"C:\Users\USER\eew\models\cnn_baseline"
os.makedirs(OUT, exist_ok=True)

SEED         = 42
EPOCHS       = 20
EARLY_STOP   = 5          
BATCH        = 64
LR           = 2e-3
NUM_WORKERS  = 0         
PRINT_EVERY  = 50        
# -------------------- DATA ----------------------
df = pd.read_csv(CSV)
y_all = (df["trace_category"].astype(str).str.lower().str.contains("earthquake")).astype(int).values
idx   = np.arange(len(df))

i_train, i_tmp, y_train, y_tmp = train_test_split(idx, y_all, test_size=0.20, random_state=SEED, stratify=y_all)
i_val,   i_test, y_val,  y_test= train_test_split(i_tmp, y_tmp,   test_size=0.50, random_state=SEED, stratify=y_tmp)

print(f"Split sizes: train={len(i_train)}  val={len(i_val)}  test={len(i_test)}")
class_counts = np.bincount(y_train, minlength=2)
weights = class_counts.sum() / (class_counts + 1e-6)
weights = (weights / weights.mean()).astype(np.float32)
print("Train class counts:", class_counts, "-> weights:", weights)

full_ds  = H5Dataset(CSV, H5)
train_ds = Subset(full_ds, i_train)
val_ds   = Subset(full_ds, i_val)
test_ds  = Subset(full_ds, i_test)

def collate(batch):
    x = torch.stack([b[0] for b in batch], dim=0)  # (B,3,128,128)
    y = torch.stack([b[1] for b in batch], dim=0)  # (B,)
    return x, y

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=NUM_WORKERS, collate_fn=collate)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate)

# -------------------- MODEL ---------------------
class TinyCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, n_classes)
    def forward(self, x):
        x = self.net(x).view(x.size(0), -1)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TinyCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# Remove deprecated verbose flag (prevents the warning)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

# -------------------- TRAIN / EVAL ----------------
def run_epoch(loader, train=True):
    model.train(train)
    n, tot_loss = 0, 0.0
    y_true, y_pred = [], []
    pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, (xb, yb) in pbar:
        xb, yb = xb.to(device), yb.to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss   = criterion(logits, yb)
        if train:
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * xb.size(0)
        n += xb.size(0)

        preds = logits.argmax(1)
        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())

        if i % PRINT_EVERY == 0:
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(f'{"train" if train else "valid"} | loss {loss.item():.4f} | lr {lr:.2e}')

    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return tot_loss/n, acc, p, r, f1

history = {
    "epoch": [], "lr": [], "train_loss": [], "train_acc": [], "train_p": [], "train_r": [], "train_f1": [],
    "val_loss": [], "val_acc": [], "val_p": [], "val_r": [], "val_f1": []
}

best_val = -1.0
best_ep  = 0
best_path = os.path.join(OUT, "cnn_baseline.pt")
log_csv   = os.path.join(OUT, "train_log.csv")

for ep in range(1, EPOCHS+1):
    t0 = time.time()
    tr_loss, tr_acc, tr_p, tr_r, tr_f1 = run_epoch(train_loader, train=True)
    va_loss, va_acc, va_p, va_r, va_f1 = run_epoch(val_loader,   train=False)
    scheduler.step(va_loss)  # no 'verbose' => no warning
    lr_now = optimizer.param_groups[0]["lr"]

    dt = time.time() - t0
    print(f"Epoch {ep:02d}/{EPOCHS} | "
          f"train: loss={tr_loss:.4f} acc={tr_acc:.3f} p={tr_p:.3f} r={tr_r:.3f} f1={tr_f1:.3f} | "
          f"val: loss={va_loss:.4f} acc={va_acc:.3f} p={va_p:.3f} r={va_r:.3f} f1={va_f1:.3f} | "
          f"lr={lr_now:.2e} | {dt:.1f}s")

    # log
    history["epoch"].append(ep); history["lr"].append(lr_now)
    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
    history["train_p"].append(tr_p); history["train_r"].append(tr_r); history["train_f1"].append(tr_f1)
    history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
    history["val_p"].append(va_p); history["val_r"].append(va_r); history["val_f1"].append(va_f1)
    pd.DataFrame(history).to_csv(log_csv, index=False)

    # checkpoint + early stop
    if va_acc > best_val:
        best_val = va_acc
        best_ep  = ep
        torch.save(model.state_dict(), best_path)
        print("  -> saved best:", best_path)
    if ep - best_ep >= EARLY_STOP:
        print(f"Early stopping (no val improvement for {EARLY_STOP} epochs). Best epoch was {best_ep}.")
        break

# -------------------- TEST -----------------------
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in tqdm(test_loader, total=len(test_loader), leave=False, desc="test"):
        logits = model(xb.to(device))
        y_true += yb.numpy().tolist()
        y_pred += logits.argmax(1).cpu().numpy().tolist()

report = classification_report(y_true, y_pred, target_names=["noise","earthquake"], digits=3)
cm     = confusion_matrix(y_true, y_pred).tolist()
print("\nTEST REPORT\n", report)

with open(os.path.join(OUT,"metrics.json"), "w") as f:
    json.dump({
        "report": report,
        "confusion_matrix": cm,
        "labels": ["noise","earthquake"],
        "best_val_acc": best_val
    }, f, indent=2)

print("\nSaved:")
print(" - best model:", best_path)
print(" - logs CSV  :", log_csv)
print(" - metrics   :", os.path.join(OUT, "metrics.json"))

