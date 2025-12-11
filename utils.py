# utils.py
import numpy as np
from scipy.signal import butter, filtfilt

# ============================================================
# Utility functions
# ============================================================

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def eye_aspect_ratio(pts):
    """
    pts: array of shape (6, 2) for an eye:
    p1-p6 as in the EAR formula.
    """
    p1, p2, p3, p4, p5, p6 = pts

    def dist(a, b):
        return np.linalg.norm(a - b)

    vertical = dist(p2, p6) + dist(p3, p5)
    horizontal = 2.0 * dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


# -----------------------
# NEW: lowpass helper
# -----------------------
def butter_lowpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    wn = cutoff / nyq
    return butter(order, wn, btype='low')

def lowpass_filter(data, cutoff, fs, order=3):
    """
    Zero-phase low-pass filter using filtfilt.
    data: 1D numpy array
    cutoff: cutoff frequency in Hz (e.g., 1.0 or 2.0)
    fs: sampling frequency (Hz)
    """
    if len(data) < 3:
        return data
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)
