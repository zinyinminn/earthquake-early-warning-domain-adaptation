# dataloader_h5.py — robust waveform→spectrogram + HDF5 dataset
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, spectrogram

# ---------------------- Filtering ----------------------
def bandpass_filter(x_1d: np.ndarray, fs: float, f_lo=1.0, f_hi=20.0, order=4) -> np.ndarray:
    """Apply Butterworth band-pass on a 1D array; safe for short traces."""
    if x_1d.size < 4:
        return x_1d
    nyq = 0.5 * fs
    low = max(1e-6, f_lo / nyq)
    high = min(0.999, f_hi / nyq)
    if high <= low:
        return x_1d
    b, a = butter(order, [low, high], btype="band")
    padlen = max(0, 3 * max(len(a), len(b)))
    if x_1d.size <= padlen:
        # fallback: zero-pad to avoid filtfilt error
        pad = padlen + 1 - x_1d.size
        x_1d = np.pad(x_1d, (0, pad), mode="edge")
    try:
        return filtfilt(b, a, x_1d, method="pad")
    except Exception:
        return x_1d

# ---------------------- Spectrogram ----------------------
def _resize_time(S: np.ndarray, new_t: int) -> np.ndarray:
    """Resize along time axis to new_t using linear interpolation (freq unchanged)."""
    if S.shape[1] == new_t:
        return S
    t_old = np.linspace(0.0, 1.0, S.shape[1], dtype=np.float32)
    t_new = np.linspace(0.0, 1.0, new_t, dtype=np.float32)
    out = np.empty((S.shape[0], new_t), dtype=np.float32)
    for i in range(S.shape[0]):
        out[i] = np.interp(t_new, t_old, S[i].astype(np.float32))
    return out

def wf_to_spec(x: np.ndarray,
               fs: float,
               nperseg: int = 256,
               noverlap: int = 128,
               n_freq: int = 128,
               n_time: int = 128) -> np.ndarray:
    """
    Compute per-channel spectrogram and return (C, n_freq, n_time) float32.
    - Accepts x as (C,T) or (T,C) and standardizes to (C,T).
    - Pads to at least nperseg to avoid spectrogram errors.
    """
    # standardize to (C,T)
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"wf_to_spec expects 2D array, got shape {x.shape}")
    # If first dim looks like time (>> channels), transpose
    if x.shape[0] > x.shape[1]:
        # Could be (T,C); detect typical 6000x3 and swap
        if x.shape[1] <= 16:
            x = x.T  # -> (C,T)
    # If still (T,C), fix again
    if x.shape[0] in (1,2,3) and x.shape[1] > 16:
        C, T = x.shape
    else:
        # ambiguous: fallback – assume first is channels
        C, T = x.shape

    # ensure minimum length for spectrogram window
    if T < nperseg:
        pad = nperseg - T
        x = np.pad(x, ((0,0),(0,pad)), mode="edge")
        T = x.shape[1]

    specs = []
    for c in range(x.shape[0]):
        # magnitude spectrogram
        f, t, Sxx = spectrogram(
            x[c].astype(np.float32),
            fs=fs,
            nperseg=nperseg,
            noverlap=min(noverlap, max(0, nperseg - 1)),
            nfft=nperseg,
            detrend=False,
            scaling="density",
            mode="magnitude"
        )  # Sxx: (F, Tt)
        S = np.log1p(Sxx).astype(np.float32)   # log magnitude

        # take lowest n_freq bins (EEW useful band) or pad if needed
        if S.shape[0] >= n_freq:
            S = S[:n_freq, :]
        else:
            pad_f = n_freq - S.shape[0]
            S = np.pad(S, ((0, pad_f), (0, 0)), mode="edge")

        # resize time axis to n_time
        S = _resize_time(S, n_time)

        # per-channel z-score then clip (stable for cross-datasets)
        mu = float(np.mean(S))
        sd = float(np.std(S) + 1e-6)
        S = (S - mu) / sd
        S = np.clip(S, -4.0, 4.0).astype(np.float32)
        specs.append(S)

    return np.stack(specs, axis=0).astype(np.float32)  # (C, F, T)

# ---------------------- Dataset ----------------------
class H5Dataset(Dataset):
    """
    Robust HDF5 loader for cross-dataset evaluation.
    - mode: "strict" → returns (3,128,128) spectrograms
            "raw"    → returns normalized waveform (C,T) float32
    - z_only: pick a single 'vertical' channel and replicate to 3 for the classifier
    - bandpass: (f_lo, f_hi) in Hz to filter waveform before spectrogram
    - min_len: discard traces shorter than this (unless mode pads them)
    - return_name: also return the HDF5 key / trace name
    """
    def __init__(self,
                 csv_file: str,
                 h5_file: str,
                 mode: str = "strict",
                 z_only: bool = False,
                 bandpass=None,
                 limit: int = 0,
                 min_len: int = 10,
                 return_name: bool = False):
        self.csv_file = csv_file
        self.h5_file  = h5_file
        self.mode     = mode
        self.z_only   = z_only
        self.bandpass = bandpass
        self.limit    = limit
        self.min_len  = max(0, int(min_len))
        self.return_name = return_name

        with h5py.File(self.h5_file, "r") as f:
            # Accept either: f['data'][key] or f[key]
            if "data" in f and isinstance(f["data"], h5py.Group):
                self.root = "data"
                keys = list(f["data"].keys())
            else:
                self.root = None
                keys = list(f.keys())

        # Filter too-short waveforms if min_len>0
        self.keys = []
        with h5py.File(self.h5_file, "r") as f:
            for k in keys:
                obj = f[self.root][k] if self.root else f[k]
                # group → dataset at "data"; dataset → use directly
                arr = np.array(obj["data"]) if isinstance(obj, h5py.Group) else np.array(obj)
                # shape can be (T,C) or (C,T)
                T = arr.shape[0] if arr.shape[0] > arr.shape[1] else arr.shape[1]
                if T >= self.min_len:
                    self.keys.append(k)

        if self.limit > 0:
            self.keys = self.keys[: self.limit]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        name = self.keys[idx]
        with h5py.File(self.h5_file, "r") as f:
            obj = f[self.root][name] if self.root else f[name]
            if isinstance(obj, h5py.Group):
                x = np.array(obj["data"])  # could be (T,C) or (C,T)
                y = int(obj.attrs.get("label_eq", 0))
                fs = float(obj.attrs.get("fs", 100.0))
            else:
                x = np.array(obj)
                # attrs may be on parent group or absent → default
                y = 0
                fs = 100.0

        # standardize orientation to (C,T)
        x = np.asarray(x)
        if x.ndim != 2:
            # fallback: flatten → (1,T)
            x = x.reshape(1, -1)
        if x.shape[0] > x.shape[1] and x.shape[1] <= 16:
            x = x.T  # (C,T)
        if x.shape[0] not in (1,2,3) and x.shape[1] in (1,2,3):
            x = x.T

        # Apply band-pass
        if self.bandpass is not None:
            f_lo, f_hi = self.bandpass
            for c in range(x.shape[0]):
                x[c] = bandpass_filter(x[c], fs, f_lo, f_hi)

        # Normalize waveform per channel (for raw mode and for spec stability)
        x = x.astype(np.float32)
        mu = np.mean(x, axis=1, keepdims=True)
        sd = np.std(x, axis=1, keepdims=True) + 1e-6
        x = (x - mu) / sd

        if self.mode == "raw":
            X = x  # (C,T) float32
        else:
            # Spectrogram (C,128,128)
            X = wf_to_spec(x, fs=fs, nperseg=256, noverlap=128, n_freq=128, n_time=128)

        # z-only? pick a vertical-like channel index (prefer last if >=3)
        if self.z_only:
            z_idx = 2 if X.shape[0] >= 3 else 0
            Z = X[z_idx:z_idx+1, :, :]
            # replicate to 3 channels for the classifier
            X = np.repeat(Z, 3, axis=0)

        # to tensor
        X = torch.from_numpy(X.astype(np.float32))

        if self.return_name:
            return X, y, name
        return X, y
