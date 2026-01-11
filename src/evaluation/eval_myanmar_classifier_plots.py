# eval_myanmar_classifier_plots.py
#
# Evaluate Myanmar classifier + generate thesis plots:
#   - confusion matrix
#   - ROC curve
#   - PR curve
#   - threshold vs F1
#   - metrics JSON

import os
import json
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    classification_report,
)


CSV_PATH = r"D:\datasets\myanmar_eq\myanmar_full.csv"
H5_PATH  = r"D:\datasets\myanmar_eq\myanmar_full.hdf5"
CKPT     = r"D:\datasets\eew\models\myanmar_cnn_improved\cnn_myanmar_ft_improved.pt"
OUT_DIR  = r"D:\datasets\myanmar_eq\thesis_figures"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Utility: spectrogram function (same style as STEAD/USGS)
# ---------------------------------------------------------------------
from dataloader_h5 import wf_to_spec  # already exists in your project

FS = 100.0  # Hz


class MyanmarFullDataset(Dataset):
    """
    Classification dataset for Myanmar full H5.

    CSV:   myanmar_full.csv  (trace_name, label_eq, ...)
    HDF5:  myanmar_full.hdf5  (one dataset per trace_name, waveform (C,T))
    """

    def __init__(self, csv_path, h5_path):
        self.csv_path = csv_path
        self.h5_path = h5_path

        df = pd.read_csv(self.csv_path)
        if "trace_name" not in df.columns or "label_eq" not in df.columns:
            raise ValueError("CSV must contain 'trace_name' and 'label_eq' columns")

        self.names = df["trace_name"].astype(str).tolist()
        self.labels = df["label_eq"].astype(int).to_numpy()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        y = self.labels[idx]

        with h5py.File(self.h5_path, "r") as f:
            if name not in f:
                raise KeyError(f"Trace {name} not found in H5 file")
            x = f[name][()]  # (C,T) float32

        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[0] > x.shape[1]:
            x = x.T  # (C,T)

        # per-channel z-score
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-6
        x = (x - mu) / sd

        # spectrogram (C, 128, 128)
        spec = wf_to_spec(
            x,
            fs=FS,
            nperseg=256,
            noverlap=128,
            n_freq=128,
            n_time=128,
        ).astype(np.float32)

        X = torch.from_numpy(spec)
        y = torch.tensor(y, dtype=torch.long)
        return X, y


def collate(batch):
    xb = torch.stack([b[0] for b in batch], dim=0)
    yb = torch.stack([b[1] for b in batch], dim=0)
    return xb, yb



class MyanmarCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # indices 0–3: first block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 0 conv  3 -> 32
            nn.BatchNorm2d(32),                          # 1 bn32
            nn.ReLU(inplace=True),                       # 2
            nn.MaxPool2d(2, 2),                          # 3

            # index 4: dummy to shift conv64 to index 5
            nn.Identity(),                               # 4 (no params)

            # second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 5 conv 32 -> 64
            nn.BatchNorm2d(64),                          # 6 bn64
            nn.ReLU(inplace=True),                       # 7
            nn.MaxPool2d(2, 2),                          # 8

            # index 9: dummy to shift conv128 to index 10
            nn.Identity(),                               # 9 (no params)

            # third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),#10 conv 64 -> 128
            nn.BatchNorm2d(128),                         #11 bn128
            nn.ReLU(inplace=True),                       #12
            nn.AdaptiveAvgPool2d((1, 1)),                #13
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),   # matches classifier.1.weight [64,128]
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



def main():
    print("==============================================")
    print("MYANMAR CLASSIFIER EVALUATION + PLOTS")
    print("==============================================")
    print(f"[INFO] CSV : {CSV_PATH}")
    print(f"[INFO] H5  : {H5_PATH}")
    print(f"[INFO] CKPT: {CKPT}")
    print(f"[INFO] OUT : {OUT_DIR}")

    ds = MyanmarFullDataset(CSV_PATH, H5_PATH)
    print(f"[INFO] Dataset size: {len(ds)} samples")

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----- load model -----
    model = MyanmarCNN(n_classes=2).to(device)

    ckpt = torch.load(CKPT, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    # allow a few harmless extra keys if any
    model.load_state_dict(state, strict=False)
    print("✓ Loaded classifier weights from:", CKPT)

    # ----- inference -----
    all_y_true = []
    all_y_score = []  # P(eq)

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_y_true.append(yb.numpy())
            all_y_score.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y_true).astype(int)
    y_score = np.concatenate(all_y_score).astype(float)

    # baseline threshold
    y_pred_05 = (y_score >= 0.5).astype(int)

    acc_05 = accuracy_score(y_true, y_pred_05)
    f1_05 = f1_score(y_true, y_pred_05)
    prec_05 = precision_score(y_true, y_pred_05)
    rec_05 = recall_score(y_true, y_pred_05)
    cm_05 = confusion_matrix(y_true, y_pred_05)

    print("\nClassification report (th=0.5):")
    print(
        classification_report(
            y_true,
            y_pred_05,
            target_names=["Noise", "Earthquake"],
            digits=3,
        )
    )
    print(f"Accuracy : {acc_05:.4f}")
    print(f"F1 score : {f1_05:.4f}")
    print(f"Precision: {prec_05:.4f}")
    print(f"Recall   : {rec_05:.4f}")
    print("Confusion matrix:\n", cm_05)

    # ROC / PR
    fpr, tpr, roc_th = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    prec_curve, rec_curve, pr_th = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    # threshold sweep (for F1 + accuracy)
    ths = np.linspace(0.0, 1.0, 201)
    f1s, accs = [], []
    for th in ths:
        yp = (y_score >= th).astype(int)
        f1s.append(f1_score(y_true, yp))
        accs.append(accuracy_score(y_true, yp))
    f1s = np.array(f1s)
    accs = np.array(accs)
    best_idx = int(np.argmax(f1s))
    best_th = float(ths[best_idx])
    best_f1 = float(f1s[best_idx])
    best_acc = float(accs[best_idx])
    print(f"\n[THRESHOLD] Best F1 at th={best_th:.3f}: F1={best_f1:.4f}, Acc={best_acc:.4f}")

    # metrics JSON
    metrics = {
        "n_samples": int(len(y_true)),
        "n_noise": int((y_true == 0).sum()),
        "n_eq": int((y_true == 1).sum()),
        "acc_05": float(acc_05),
        "f1_05": float(f1_05),
        "precision_05": float(prec_05),
        "recall_05": float(rec_05),
        "cm_05": cm_05.tolist(),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "best_threshold": best_th,
        "best_f1": best_f1,
        "best_acc": best_acc,
    }

    with open(os.path.join(OUT_DIR, "myanmar_classifier_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("[INFO] Saved metrics JSON")

    # --------------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------------

    # 1) Confusion matrix (th=0.5)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm_05, cmap="Blues")
    plt.title(f"Myanmar classifier (th=0.5)\nAcc={acc_05:.3f}, F1={f1_05:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Noise", "EQ"])
    plt.yticks([0, 1], ["Noise", "EQ"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm_05[i, j], ha="center", va="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mm_confusion_matrix.png"), dpi=150)
    plt.close()

    # 2) ROC curve
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Myanmar classifier ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mm_roc_curve.png"), dpi=150)
    plt.close()

    # 3) PR curve
    plt.figure(figsize=(5, 5))
    plt.plot(rec_curve, prec_curve, label=f"AP={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Myanmar classifier PR curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mm_pr_curve.png"), dpi=150)
    plt.close()

    # 4) Threshold vs F1 + Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(ths, f1s, label="F1")
    plt.plot(ths, accs, label="Accuracy", linestyle="--")
    plt.axvline(best_th, color="k", linestyle=":", label=f"best th={best_th:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Myanmar classifier – threshold tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mm_threshold_f1.png"), dpi=150)
    plt.close()

    print("[INFO] Saved all classification plots to:", OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()

