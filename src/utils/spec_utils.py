# spec_utils.py
import numpy as np
from scipy import signal


def make_spec_from_waveform(x, fs=100, size=128, **kwargs):
    """
    Convert waveform to a 3-channel spectrogram (3, size, size).

    This keeps your original behaviour:
    - nperseg = 129
    - scaling = 'density'
    - mode = 'magnitude'
    - log1p
    - per-channel normalization

    Backward compatible:
    - make_spec_from_waveform(x)
    - make_spec_from_waveform(x, fs=100)
    - make_spec_from_waveform(x, sampling_rate=100)
    """

    
    if "sampling_rate" in kwargs and kwargs["sampling_rate"] is not None:
        try:
            fs = float(kwargs["sampling_rate"])
        except Exception:
            # if conversion fails, just keep original fs
            pass

    # ---- core logic (robust) ----
    x = np.nan_to_num(np.array(x, dtype=np.float32))


    if x.ndim == 1:
        # (T,) -> (T,1)
        x = x[:, None]

    # If it's (3, T) (common for mseed), transpose to (T, 3)
    if x.shape[0] == 3 and x.shape[1] > 3:
        x = x.T

    # Now x is (T, C). Ensure exactly 3 channels.
    if x.shape[1] < 3:
        reps = int(np.ceil(3 / x.shape[1]))
        x = np.tile(x, (1, reps))[:, :3]
    elif x.shape[1] > 3:
        x = x[:, :3]

    # per-channel mean removal 
    x = x - x.mean(axis=0, keepdims=True)

   
    base_nperseg = 129
    base_noverlap = base_nperseg 
    mode     = "magnitude"
    scaling  = "density"
    detrend  = "constant"
    norm     = "per_channel"  

    chans = []
    for ch in range(3):
        sig = x[:, ch]

        # guard extremely short signals
        if len(sig) < 2:
            # produce a zero spectrogram instead of crashing
            chans.append(np.zeros((size, size), np.float32))
            continue

        # adapt nperseg / noverlap for short signals
        nperseg = min(base_nperseg, len(sig))
        if nperseg <= 1:
            chans.append(np.zeros((size, size), np.float32))
            continue
        noverlap = min(base_noverlap, nperseg - 1)

        f, t, Sxx = signal.spectrogram(
            sig,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling=scaling,
            mode=mode,
        )
        if mode != "magnitude":
            Sxx = np.sqrt(Sxx + 1e-12)
        Sxx = np.log1p(Sxx).astype(np.float32)

        out = np.zeros((size, size), np.float32)
        h = min(size, Sxx.shape[0])
        w = min(size, Sxx.shape[1])
        out[:h, :w] = Sxx[:h, :w]
        chans.append(out)

    spec = np.stack(chans, axis=0)

    # same normalization choices as your original file
    if norm == "global":
        m, s = spec.mean(), spec.std() + 1e-6
        spec = (spec - m) / s
    else:
        m = spec.mean(axis=(1, 2), keepdims=True)
        s = spec.std(axis=(1, 2), keepdims=True) + 1e-6
        spec = (spec - m) / s

    return spec

