# rppg_deepphys.py
import time
from collections import deque
import numpy as np
import cv2
import torch

from deepphys_loader import load_deepphys
from utils import bandpass_filter

class DeepPhysEstimator:
    """
    Wrapper for DeepPhys checkpoint.
    Usage:
      r = DeepPhysEstimator(weights_path="weights/UBFC-rPPG_DeepPhys.pth", window_seconds=20)
      r.update(frame, face_landmarks)
      hr, quality = r.estimate_hr_and_quality()
    """

    def __init__(self, weights_path, window_seconds=20.0, img_size=36, min_interval=0.5, device=None):
        self.window_seconds = window_seconds
        self.img_size = img_size  # DeepPhys was trained on small imgs (36)
        self.min_interval = float(min_interval)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.device = load_deepphys(weights_path, device=self.device)

        # sliding buffers
        self.times = deque()
        self.rppg_vals = deque()  # model outputs (one scalar per processed frame)
        self.prev_frame = None

        # last computed HR / quality
        self.last_hr = None
        self.last_quality = 0.0
        self.last_infer_time = 0.0

    def _compute_bbox(self, frame, face_landmarks):
        h, w, _ = frame.shape
        xs = [lm.x * w for lm in face_landmarks]
        ys = [lm.y * h for lm in face_landmarks]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        # small padding
        pad_x = int(0.08 * (x_max - x_min))
        pad_y = int(0.08 * (y_max - y_min))
        x_min = max(0, x_min - pad_x); x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y); y_max = min(h, y_max + pad_y)
        if x_max <= x_min or y_max <= y_min:
            return None
        return x_min, x_max, y_min, y_max

    def _prepare_input(self, crop, prev_crop):
        """
        DeepPhys expects input channels: [diff_rgb (3), raw_rgb (3)] as a 6-channel 2D input.
        We create a tensor shaped (1,6,H,W) normalized to 0..1 floats.
        crop, prev_crop: uint8 BGR images (H,W,3)
        """
        # convert BGR->RGB and to float32 [0,1]
        raw = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if prev_crop is None:
            diff = np.zeros_like(raw)
        else:
            prev_raw = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            diff = raw - prev_raw  # difference emphasizes motion
        # stack channels: first diff, then raw
        stacked = np.concatenate([diff, raw], axis=2)  # (H, W, 6)
        # transpose to (C, H, W)
        tensor = np.transpose(stacked, (2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)  # (1, 6, H, W)
        return tensor

    def update(self, frame, face_landmarks):
        """
        Call each frame. This will compute one DeepPhys scalar output per frame (after model forward),
        append it to the internal rppg buffer and keep timestamps.
        """
        bbox = self._compute_bbox(frame, face_landmarks)
        if bbox is None:
            return
        x_min, x_max, y_min, y_max = bbox
        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return

        # resize to model's img_size
        try:
            crop_resized = cv2.resize(crop, (self.img_size, self.img_size))
        except Exception:
            return

        # Prepare input and run model (throttled by min_interval to avoid too many forwards)
        now = time.time()
        # We run model every frame (DeepPhys is fairly light). If need be, throttle by min_interval.
        if now - self.last_infer_time < self.min_interval:
            # Even if skipping forward, still push last model output or 0 placeholder
            # For temporal alignment, append last rppg value (or 0)
            val = float(self.rppg_vals[-1]) if len(self.rppg_vals) else 0.0
            self.times.append(now)
            self.rppg_vals.append(val)
        else:
            inp = self._prepare_input(crop_resized, self.prev_frame)
            with torch.no_grad():
                out = self.model(inp)  # model returns (batch,1)
            # out might be shape (1,1) or (1,)
            val = out.squeeze().cpu().item()
            self.times.append(now)
            self.rppg_vals.append(float(val))
            self.last_infer_time = now

        # keep prev_frame for next diff
        self.prev_frame = crop_resized.copy()

        # trim buffer to window_seconds
        while len(self.times) > 2 and (self.times[-1] - self.times[0] > self.window_seconds):
            self.times.popleft()
            self.rppg_vals.popleft()

    def estimate_hr_and_quality(self):
        """
        Post-process recent DeepPhys scalar outputs (rppg_vals) to estimate HR and quality.
        """
        if len(self.times) < 30:
            return None, 0.0

        t = np.array(self.times)
        s = np.array(self.rppg_vals, dtype=float)

        duration = t[-1] - t[0]
        if duration < 5.0:
            return None, 0.0

        # Detrend / normalize
        s = s - np.mean(s)
        std = np.std(s)
        if std < 1e-6:
            return None, 0.0
        s = s / std

        # Estimate sampling rate
        fs = len(t) / duration

        # Bandpass filter in physiological band
        lowcut, highcut = 0.75, 3.0
        try:
            s_f = bandpass_filter(s, lowcut, highcut, fs, order=3)
        except Exception:
            return None, 0.0

        # FFT -> peak frequency
        n = len(s_f)
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        spectrum = np.abs(np.fft.rfft(s_f))**2

        band_mask = (freqs >= lowcut) & (freqs <= highcut)
        if not np.any(band_mask):
            return None, 0.0

        freqs_band = freqs[band_mask]
        spec_band = spectrum[band_mask]
        peak_idx = int(np.argmax(spec_band))
        f_peak = float(freqs_band[peak_idx])
        hr = 60.0 * f_peak

        # SNR-like quality
        peak_power = float(spec_band[peak_idx])
        rest_power = float((np.sum(spec_band) - peak_power) / max(len(spec_band)-1, 1))
        quality = 0.0
        if rest_power > 0:
            ratio = peak_power / rest_power
            quality = min(ratio, 5.0) / 5.0

        self.last_hr = float(hr)
        self.last_quality = float(quality)
        return self.last_hr, self.last_quality
