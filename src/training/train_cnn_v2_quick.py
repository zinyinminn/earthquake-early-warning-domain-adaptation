import os, json, time, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import sys
sys.path.insert(0, r"C:\Users\USER\eew\scripts")
from dataloader_h5 import H5Dataset

CSV = r"D:\datasets\stead_subset\subset.csv"
H5  = r"D:\datasets\stead_subset\subset.hdf5"
OUT = r"C:\Users\USER\eew\models\cnn_v2"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
y_all = (df["trace_category"].astype(str).str.lower().str.contains("earthquake")).astype(int).values
idx = np.arange(len(df))
i_tr, i_tmp, y_tr, y_tmp = train_test_split(idx, y_all, test_size=0.20, random_state=42, stratify=y_all)
i_va, i_te, y_va, y_te = train_test_split(i_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

ds = H5Dataset(CSV, H5)
def collate(b):
    x = torch.stack([t[0] for t in b]); y = torch.stack([t[1] for t in b])
    return x, y
train_loader = DataLoader(Subset(ds, i_tr), batch_size=64, shuffle=True, collate_fn=collate)
val_loader   = DataLoader(Subset(ds, i_va), batch_size=64, shuffle=False, collate_fn=collate)
test_loader  = DataLoader(Subset(ds, i_te), batch_size=64, shuffle=False, collate_fn=collate)

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        x = self.net(x); x = x.view(x.size(0), -1); return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCNN().to(device)
class_counts = np.bincount(y_tr, minlength=2)
weights = class_counts.sum() / (class_counts + 1e-6); weights = (weights/weights.mean()).astype(np.float32)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)

def run(loader, train):
    model.train(train); n, tot, correct = 0,0.0,0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        if train: opt.zero_grad()
        lg = model(xb); loss = criterion(lg,yb)
        if train: loss.backward(); opt.step()
        tot += loss.item()*xb.size(0); correct += (lg.argmax(1)==yb).sum().item(); n+=xb.size(0)
    return tot/n, correct/n

best, BEST_PATH = -1.0, os.path.join(OUT,"cnn_v2.pt")
for ep in range(1, 7):  # 6-7 quick epochs
    trL,trA = run(train_loader, True)
    vaL,vaA = run(val_loader, False)
    sched.step(vaL)
    print(f"Epoch {ep:02d} | train {trL:.3f}/{trA:.3f} | val {vaL:.3f}/{vaA:.3f}")
    if vaA > best:
        best = vaA; torch.save(model.state_dict(), BEST_PATH); print("  -> saved", BEST_PATH)

# test
model.load_state_dict(torch.load(BEST_PATH, map_location=device))
model.eval(); y_t, y_p = [], []
with torch.no_grad():
    for xb,yb in test_loader:
        pr = model(xb.to(device)).argmax(1).cpu().numpy()
        y_p.append(pr); y_t.append(yb.numpy())
y_t = np.concatenate(y_t); y_p = np.concatenate(y_p)
print("TEST ACC:", accuracy_score(y_t,y_p))
print(classification_report(y_t,y_p, target_names=["noise","earthquake"], digits=3))
