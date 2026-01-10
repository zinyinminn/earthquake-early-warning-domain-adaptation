# train_cnn_balanced.py
import os, json, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from dataloader_h5 import H5Dataset_Cls, wf_to_spec   # your existing dataset

# ========= PATHS =========
CSV = r"D:\datasets\stead_subset\subset.csv"
H5  = r"D:\datasets\stead_subset\subset.hdf5"
OUT_DIR = r"C:\Users\USER\eew\models\cnn_balanced"
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUT_DIR, "cnn_balanced.pt")
LOG_CSV    = os.path.join(OUT_DIR, "train_log.csv")

# ========= HYPERPARAMS =========
EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
PATIENCE = 5

# ========= MODEL =========
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64,2)
    def forward(self,x):
        z = self.net(x).view(x.size(0),-1)
        return self.fc(z)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCNN().to(device)

# ========= DATA =========
train_ds = H5Dataset_Cls(CSV,H5,split="train")
val_ds   = H5Dataset_Cls(CSV,H5,split="val")
test_ds  = H5Dataset_Cls(CSV,H5,split="test")

# Weighted sampler for training (balance noise/eq)
class_counts = np.bincount(train_ds.df["label"].values)
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[train_ds.df["label"].values]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_ld  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ========= OPTIMIZER =========
opt = optim.Adam(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

# ========= TRAIN LOOP =========
history = []
best_val_loss, patience_counter = 1e9, 0

def run_epoch(loader, train):
    total_loss, correct, n = 0,0,0
    y_true,y_pred=[],[]
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            out = model(xb)
            loss = crit(out,yb)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()*len(xb)
        pred = out.argmax(1)
        correct += (pred==yb).sum().item()
        n += len(xb)
        y_true += yb.cpu().numpy().tolist()
        y_pred += pred.cpu().numpy().tolist()
    acc = correct/n
    rep = classification_report(y_true,y_pred,output_dict=True,zero_division=0)
    return total_loss/n, acc, rep

for epoch in range(1,EPOCHS+1):
    tr_loss,tr_acc,tr_rep = run_epoch(train_ld, True)
    va_loss,va_acc,va_rep = run_epoch(val_ld, False)

    scheduler.step(va_loss)
    row = {
        "epoch":epoch,
        "tr_loss":tr_loss,"va_loss":va_loss,
        "tr_acc":tr_acc,"va_acc":va_acc,
        "va_prec":va_rep["1"]["precision"],
        "va_rec":va_rep["1"]["recall"],
        "va_f1":va_rep["1"]["f1-score"]
    }
    history.append(row)
    print(f"Epoch {epoch}/{EPOCHS} | train loss={tr_loss:.4f} acc={tr_acc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f} f1={row['va_f1']:.3f}")

    # early stopping
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        torch.save(model.state_dict(), MODEL_PATH)
        patience_counter = 0
        print("  -> saved best model")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping")
            break

# save log
df = pd.DataFrame(history); df.to_csv(LOG_CSV,index=False)

# ========= PLOTS =========
plt.figure(); plt.plot(df["epoch"], df["tr_loss"], label="train")
plt.plot(df["epoch"], df["va_loss"], label="val"); plt.legend(); plt.title("Loss")
plt.savefig(os.path.join(OUT_DIR,"loss_curve.png"))

plt.figure(); plt.plot(df["epoch"], df["tr_acc"], label="train")
plt.plot(df["epoch"], df["va_acc"], label="val"); plt.legend(); plt.title("Accuracy")
plt.savefig(os.path.join(OUT_DIR,"acc_curve.png"))

plt.figure(); plt.plot(df["epoch"], df["va_f1"], label="val f1")
plt.legend(); plt.title("F1-score"); plt.savefig(os.path.join(OUT_DIR,"f1_curve.png"))

# ========= TEST EVAL =========
print("\n=== TEST EVALUATION ===")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
te_loss,te_acc,te_rep = run_epoch(test_ld, False)
print("Test acc:", te_acc)
print(classification_report(
    [int(x) for x in te_rep.keys() if x in ["0","1"]],
    [te_rep[k] for k in ["0","1"]],
))
cm = confusion_matrix(
    [int(y) for xb,yb in test_ld for y in yb.numpy()],
    [int(model(xb.to(device)).argmax(1).cpu()[i]) for xb,yb in test_ld for i in range(len(yb))]
)
plt.figure(); plt.imshow(cm,cmap="Blues"); plt.title("Confusion Matrix")
plt.colorbar(); plt.savefig(os.path.join(OUT_DIR,"confusion_matrix.png"))
