# server_infer.py
import torch
import h5py
import numpy as np

MODEL_PATH = "model.pt"       
DATA_PATH = "test.h5"         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

def get_real_prediction():
    """Run inference on one random test sample and return alert data."""
    with h5py.File(DATA_PATH, "r") as f:
        keys = list(f.keys())
        key = np.random.choice(keys)
        x = torch.tensor(f[key]["waveform"][:]).float().unsqueeze(0).to(DEVICE)
        lat = float(f[key].attrs.get("latitude", 0))
        lon = float(f[key].attrs.get("longitude", 0))

    with torch.no_grad():
        y = model(x).cpu().numpy().flatten()

    mag, sp, eta, dist = map(float, y[:4])
    return {
        "latitude": lat,
        "longitude": lon,
        "magnitude": mag,
        "sp_time": sp,
        "eta": eta,
        "distance": dist,
        "message": f"⚠️ Earthquake detected! M{mag:.1f}, ETA {eta:.1f}s, Dist {dist:.1f} km"
    }

if __name__ == "__main__":
    print(get_real_prediction())

