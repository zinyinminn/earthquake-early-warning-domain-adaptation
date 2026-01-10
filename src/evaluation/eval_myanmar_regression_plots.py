import os
import json
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 1. MODEL ARCHITECTURE (matches your trained checkpoint)
# ============================================================
class TinyCNNRegressor(nn.Module):
    def __init__(self, n_targets=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )

        self.reg_head = nn.Sequential(
            nn.Flatten(),            # index 0
            nn.Linear(128,64),       # index 1
            nn.ReLU(),               # index 2
            nn.Dropout(0.5),         # index 3
            nn.Linear(64,n_targets)  # index 4 (matches checkpoint)
        )

    def forward(self,x):
        return self.reg_head(self.features(x))


# ============================================================
# 2. Load spectrogram
# ============================================================
def load_spec(h5, name):
    return np.array(h5[name], dtype=np.float32)


# ============================================================
# 3. Plotting helpers
# ============================================================
def scatter_plot(true, pred, name, outdir):
    plt.figure(figsize=(7,6))
    plt.scatter(true, pred, alpha=0.5, edgecolors='k')
    lims=[min(true), max(true)]
    plt.plot(lims, lims, 'r--')
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"{name}: True vs Predicted")
    plt.grid()
    plt.savefig(os.path.join(outdir,f"{name}_scatter.png"),dpi=300)
    plt.close()

def residual_plot(true, pred, name, outdir):
    res = pred - true
    plt.figure(figsize=(7,5))
    plt.hist(res, bins=40, alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(f"{name} Residuals")
    plt.grid()
    plt.savefig(os.path.join(outdir,f"{name}_residuals.png"),dpi=300)
    plt.close()


# ============================================================
# 4. MAIN FUNCTION
# ============================================================
def main():

    CSV = r"D:\datasets\myanmar_eq\myanmar_regression.csv"
    H5_PATH = r"D:\datasets\myanmar_eq\myanmar_reg.hdf5"
    CKPT = r"D:\datasets\myanmar_eq\models_myanmar_reg\cnn_myanmar_regression_best.pt"
    OUTDIR = r"D:\datasets\myanmar_eq\thesis_figures"

    TARGETS = ["mag", "dist_km", "p_time_sec", "s_time_sec", "sp_sec"]

    print("==============================================")
    print("MYANMAR REGRESSION EVALUATION + PLOTS")
    print("==============================================")

    os.makedirs(OUTDIR, exist_ok=True)

    df = pd.read_csv(CSV)
    print(f"[INFO] Loaded {len(df)} rows")

    # Load waveform HDF5
    h5 = h5py.File(H5_PATH,"r")

    # Create model
    model = TinyCNNRegressor(n_targets=len(TARGETS))
    device = "cpu"
    model.to(device)

    # ==========================
    # Load checkpoint safely
    # ==========================
    ckpt = torch.load(CKPT, map_location=device)

    if "model_state_dict" in ckpt:
        print("[INFO] Loading checkpoint['model_state_dict']")
        state_dict = ckpt["model_state_dict"]
    else:
        print("[INFO] Loading checkpoint as raw state_dict")
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    # ==========================
    # Build arrays
    # ==========================
    X = []
    Y = []

    for _, row in df.iterrows():
        spec = load_spec(h5, row["trace_name"])
        X.append(spec)
        Y.append([row[t] for t in TARGETS])

    X = np.array(X)
    Y = np.array(Y)

    # Mean/std from CSV
    means = Y.mean(axis=0)
    stds = Y.std(axis=0)

    Y_norm = (Y - means) / stds

    # ==========================
    # Predict
    # ==========================
    preds = []

    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i]).unsqueeze(0).to(device)
            out = model(x).cpu().numpy()[0]
            preds.append(out)

    preds = np.array(preds)
    preds_real = preds * stds + means

    # ==========================
    # Metrics + plots
    # ==========================
    results = {}

    for i, name in enumerate(TARGETS):
        true = Y[:, i]
        pred = preds_real[:, i]

        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))

        results[name] = {"mae": float(mae), "rmse": float(rmse)}

        print(f"{name:10s} MAE={mae:.4f}, RMSE={rmse:.4f}")

        scatter_plot(true, pred, name, OUTDIR)
        residual_plot(true, pred, name, OUTDIR)

    with open(os.path.join(OUTDIR,"regression_eval_summary.json"),"w") as f:
        json.dump(results,f,indent=4)

    print("\n[INFO] Evaluation complete.")
    print(f"Figures saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
