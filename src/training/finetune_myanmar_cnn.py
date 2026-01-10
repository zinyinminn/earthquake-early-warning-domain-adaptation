# finetune_myanmar_cnn_fixed.py
# Fine-tune CNN on Myanmar dataset using myanmar_dataloader_h5.py
import os, time, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# Import Myanmar dataset
from myanmar_dataloader_h5 import MyanmarH5Dataset, create_myanmar_dataloaders

print("="*70)
print("MYANMAR EARTHQUAKE CNN FINE-TUNING")
print("="*70)

# -------------------- CONFIG --------------------
CSV = r"D:\datasets\myanmar_eq\myanmar_full.csv"
H5  = r"D:\datasets\myanmar_eq\myanmar_full.hdf5"
OUT = r"C:\Users\USER\eew\models\myanmar_cnn"
os.makedirs(OUT, exist_ok=True)

# Try to load pretrained STEAD model
PRETRAINED_PATH = r"C:\Users\USER\eew\models\cnn_baseline\cnn_baseline.pt"
# Alternative paths
PRETRAINED_PATHS = [
    r"C:\Users\USER\eew\models\cnn_baseline.pt",
    r"C:\Users\USER\eew\models\cnn_baseline\cnn_baseline.pt",
    r"C:\Users\USER\eew\models\usgs_cnn.pt"
]

SEED         = 42
EPOCHS       = 30
EARLY_STOP   = 5          # stop if no val improvement for N epochs
BATCH        = 64
LR           = 1e-4       # Lower LR for fine-tuning
NUM_WORKERS  = 0          # increase if disk is fast
PRINT_EVERY  = 50         # batch-interval to print LR on tqdm bar

# -------------------- DATA ----------------------
print("Creating dataloaders...")
train_loader, val_loader, test_loader, class_weights = create_myanmar_dataloaders(
    csv_path=CSV,
    h5_path=H5,
    batch_size=BATCH,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=SEED,
    mode="spec",          # Use spectrograms
    bandpass=(1.0, 20.0), # Bandpass filter
    z_only=False,         # Use all 3 channels
    return_meta=False     # Don't return metadata for training
)

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
print(f"Device: {device}")

model = TinyCNN().to(device)

# Load pretrained weights if available
pretrained_loaded = False
for pretrained_path in PRETRAINED_PATHS:
    if os.path.exists(pretrained_path):
        print(f"\nLoading pretrained weights from: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # Assume it's the state dict itself
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # Load weights with strict=False (allows partial loading)
            model.load_state_dict(new_state_dict, strict=False)
            pretrained_loaded = True
            print("✓ Pretrained weights loaded successfully")
            break
        except Exception as e:
            print(f"✗ Error loading {pretrained_path}: {e}")
            continue

if not pretrained_loaded:
    print("\n⚠️ No pretrained model found, training from scratch")

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

# -------------------- TRAIN / EVAL ----------------
def run_epoch(loader, train=True):
    model.train(train)
    n, tot_loss = 0, 0.0
    y_true, y_pred = [], []
    pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, batch in pbar:
        # Handle batch format (with/without metadata)
        if len(batch) == 3:
            xb, yb, _ = batch
        else:
            xb, yb = batch
            
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

best_val_f1 = -1.0
best_ep  = 0
best_path = os.path.join(OUT, "cnn_myanmar_ft.pt")
log_csv   = os.path.join(OUT, "train_log.csv")

print("\nStarting training...")
print("="*70)

for ep in range(1, EPOCHS+1):
    t0 = time.time()
    tr_loss, tr_acc, tr_p, tr_r, tr_f1 = run_epoch(train_loader, train=True)
    va_loss, va_acc, va_p, va_r, va_f1 = run_epoch(val_loader,   train=False)
    scheduler.step(va_loss)
    lr_now = optimizer.param_groups[0]["lr"]

    dt = time.time() - t0
    print(f"Epoch {ep:02d}/{EPOCHS} | "
          f"train: loss={tr_loss:.4f} acc={tr_acc:.3f} f1={tr_f1:.3f} | "
          f"val: loss={va_loss:.4f} acc={va_acc:.3f} f1={va_f1:.3f} | "
          f"lr={lr_now:.2e} | {dt:.1f}s")

    # log
    history["epoch"].append(ep); history["lr"].append(lr_now)
    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
    history["train_p"].append(tr_p); history["train_r"].append(tr_r); history["train_f1"].append(tr_f1)
    history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
    history["val_p"].append(va_p); history["val_r"].append(va_r); history["val_f1"].append(va_f1)
    
    # Save log every epoch
    pd.DataFrame(history).to_csv(log_csv, index=False)

    # checkpoint + early stop
    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        best_ep  = ep
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': va_loss,
            'val_f1': va_f1,
            'history': history,
            'pretrained_loaded': pretrained_loaded,
            'class_weights': class_weights.tolist()
        }, best_path)
        print(f"  → saved best model (F1: {va_f1:.4f})")
    else:
        print(f"  → no improvement (best F1: {best_val_f1:.4f} at epoch {best_ep})")
    
    # Early stopping
    if ep - best_ep >= EARLY_STOP:
        print(f"Early stopping (no val improvement for {EARLY_STOP} epochs). Best epoch was {best_ep}.")
        break

# -------------------- TEST -----------------------
print("\n" + "="*70)
print("TEST EVALUATION")
print("="*70)

# Load best model
if os.path.exists(best_path):
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
else:
    print("Warning: Best model not found, using last model")

model.eval()
y_true, y_pred, y_probs = [], [], []
with torch.no_grad():
    for batch in tqdm(test_loader, total=len(test_loader), leave=False, desc="test"):
        if len(batch) == 3:
            xb, yb, _ = batch
        else:
            xb, yb = batch
            
        logits = model(xb.to(device))
        probs = torch.softmax(logits, dim=1)
        y_true += yb.numpy().tolist()
        y_pred += logits.argmax(1).cpu().numpy().tolist()
        y_probs += probs[:, 1].cpu().numpy().tolist()  # Probability of EQ class

# Metrics
test_acc = accuracy_score(y_true, y_pred)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='binary', zero_division=0
)

print(f"\nTest Accuracy:  {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1 Score:  {test_f1:.4f}")

report = classification_report(y_true, y_pred, target_names=["noise","earthquake"], digits=4)
cm     = confusion_matrix(y_true, y_pred).tolist()
print("\nClassification Report:")
print(report)

# Save final metrics
metrics = {
    "test_accuracy": float(test_acc),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1),
    "best_val_f1": float(best_val_f1),
    "best_epoch": int(best_ep),
    "confusion_matrix": cm,
    "classification_report": classification_report(y_true, y_pred, target_names=["noise","earthquake"], output_dict=True),
    "pretrained_loaded": pretrained_loaded,
    "dataset_info": {
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "class_weights": class_weights.tolist()
    }
}

with open(os.path.join(OUT, "test_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# -------------------- CREATE PLOTS --------------------
try:
    import matplotlib.pyplot as plt
    
    # Training history plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['epoch'], history['train_loss'], 'b-', label='Train', linewidth=2)
    plt.plot(history['epoch'], history['val_loss'], 'r-', label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['epoch'], history['train_acc'], 'b-', label='Train', linewidth=2)
    plt.plot(history['epoch'], history['val_acc'], 'r-', label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['epoch'], history['train_f1'], 'b-', label='Train', linewidth=2)
    plt.plot(history['epoch'], history['val_f1'], 'r-', label='Val', linewidth=2)
    plt.axhline(y=best_val_f1, color='g', linestyle='--', label=f'Best F1: {best_val_f1:.3f}')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUT, "myanmar_training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Myanmar Test Set (F1={test_f1:.3f})', fontsize=14)
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Noise', 'Earthquake'], rotation=45)
    plt.yticks(tick_marks, ['Noise', 'Earthquake'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Add text annotations
    thresh = np.array(cm).max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i][j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "myanmar_confusion_matrix.png"), dpi=300)
    plt.close()
    
    print(f"✓ Training plots saved to: {OUT}")
    
except Exception as e:
    print(f"Could not generate plots: {e}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Best model saved: {best_path}")
print(f"Test F1 Score:    {test_f1:.4f}")
print(f"Training log:     {log_csv}")
print(f"Test metrics:     {os.path.join(OUT, 'test_metrics.json')}")
print(f"\n✅ Myanmar CNN model ready for use in EEW demo!")