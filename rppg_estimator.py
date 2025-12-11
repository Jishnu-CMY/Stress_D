# rppg_estimator.py
import time
from collections import deque
import numpy as np
import cv2
import torch
from physnet_model import load_physnet_model
from utils import bandpass_filter


class RPPGEstimator:
    """
    PhysNet-based rPPG estimator with throttled inference.

    - update(): called every frame, only does cropping + buffering (fast).
    - estimate_hr_and_quality():
        * runs PhysNet at most once every `min_interval` seconds,
          otherwise returns the last HR/quality (cached).
    """

    def __init__(
        self,
        window_frames=128,         # temporal length PhysNet expects
        target_fps=30.0,
        device=None,
        weights_path=None,         # path to PhysNet .pth
        min_interval=1.0,          # minimum seconds between PhysNet calls
        bbox_ema_alpha=0.7        
    ):
        self.window_frames = window_frames
        self.target_fps = target_fps
        self.min_interval = min_interval

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_physnet_model(
            weights_path=weights_path,
            frames=window_frames,
            device=self.device,
        )

        # Buffers
        self.frames = deque()
        self.times = deque()

        # Input size for PhysNet
        self.input_height = 128
        self.input_width = 128

        # Cached outputs
        self.last_hr = None
        self.last_quality = 0.0
        self.last_run_time = 0.0

        self.prev_bbox = None  # (x_min,x_max,y_min,y_max) in pixel coords
        self.bbox_ema_alpha = bbox_ema_alpha

    # ----------------- helpers -----------------

    def _compute_bbox_from_landmarks(self, frame, face_landmarks):
        h, w, _ = frame.shape
        xs = [lm.x * w for lm in face_landmarks]
        ys = [lm.y * h for lm in face_landmarks]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        # add small padding
        pad_x = int(0.1 * (x_max - x_min))
        pad_y = int(0.1 * (y_max - y_min))
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        return (x_min, x_max, y_min, y_max)

    def _smoothed_bbox(self, bbox):
        """Apply EMA to the bounding box to stabilize it across frames."""
        if self.prev_bbox is None:
            self.prev_bbox = bbox
            return bbox
        a = self.bbox_ema_alpha
        x_min = int(a * bbox[0] + (1 - a) * self.prev_bbox[0])
        x_max = int(a * bbox[1] + (1 - a) * self.prev_bbox[1])
        y_min = int(a * bbox[2] + (1 - a) * self.prev_bbox[2])
        y_max = int(a * bbox[3] + (1 - a) * self.prev_bbox[3])
        sm = (x_min, x_max, y_min, y_max)
        self.prev_bbox = sm
        return sm

    def _extract_face_crop(self, frame, face_landmarks):
        bbox = self._compute_bbox_from_landmarks(frame, face_landmarks)
        bbox = self._smoothed_bbox(bbox)
        x_min, x_max, y_min, y_max = bbox
        if x_max <= x_min or y_max <= y_min:
            return None
        face_crop = frame[y_min:y_max, x_min:x_max]
        face_crop = cv2.resize(face_crop, (self.input_width, self.input_height))
        # Optional: apply mild histogram equalization on Y channel to reduce lighting variation
        try:
            ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y = cv2.equalizeHist(y)
            face_crop = cv2.merge([y, cr, cb])
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_YCrCb2BGR)
        except Exception:
            # keep original if conversion fails
            pass
        return face_crop

    def _run_physnet(self):
        if len(self.frames) < self.window_frames:
            return None, None

        frames_list = list(self.frames)[-self.window_frames :]
        times_list = list(self.times)[-self.window_frames :]

        frames_np = np.stack(frames_list, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
        frames_np = np.transpose(frames_np, (3, 0, 1, 2))                     # (C, T, H, W)
        frames_t = torch.from_numpy(frames_np).unsqueeze(0).to(self.device)   # (1, C, T, H, W)

        with torch.no_grad():
            rPPG, _, _, _ = self.model(frames_t)  # [1, frames]
        rPPG = rPPG.squeeze(0).cpu().numpy()      # (frames,)

        t = np.array(times_list)
        duration = t[-1] - t[0]
        if duration <= 0:
            return None, None
        fs = len(rPPG) / duration

        return rPPG, fs

    # ----------------- public API -----------------

    def update(self, frame, face_landmarks):
        crop = self._extract_face_crop(frame, face_landmarks)
        if crop is None:
            return

        t = time.time()
        self.frames.append(crop)
        self.times.append(t)

        # Prevent unbounded growth: keep at most 2*window_frames
        while len(self.frames) > 2 * self.window_frames:
            self.frames.popleft()
            self.times.popleft()

    def estimate_hr_and_quality(self):
        """
        Throttled PhysNet call:
        - If called again within `min_interval` seconds since last run,
          simply return the previous HR/quality.
        - Otherwise, run PhysNet on current buffer and update the cache.
        """
        now = time.time()

        # Not enough data yet
        if len(self.frames) < self.window_frames:
            return self.last_hr, self.last_quality

        # If last PhysNet run was recent, reuse cached values
        if now - self.last_run_time < self.min_interval:
            return self.last_hr, self.last_quality

        # --- PhysNet inference ---
        rppg, fs = self._run_physnet()
        if rppg is None or fs is None:
            return self.last_hr, self.last_quality

        # Detrend & normalize
        r = rppg - np.mean(rppg)
        std = np.std(r)
        if std < 1e-6:
            return self.last_hr, self.last_quality
        r = r / std

        # Optional: band-pass for robustness
        lowcut, highcut = 0.7, 3.0
        try:
            r_f = bandpass_filter(r, lowcut, highcut, fs, order=3)
        except ValueError:
            return self.last_hr, self.last_quality

        # FFT → HR
        n = len(r_f)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spectrum = np.abs(np.fft.rfft(r_f)) ** 2

        band_mask = (freqs >= lowcut) & (freqs <= highcut)
        if not np.any(band_mask):
            return self.last_hr, self.last_quality

        freqs_band = freqs[band_mask]
        spec_band = spectrum[band_mask]

        peak_idx = np.argmax(spec_band)
        f_peak = freqs_band[peak_idx]
        hr = 60.0 * f_peak  # bpm

        # Quality
        peak_power = spec_band[peak_idx]
        rest_power = (np.sum(spec_band) - peak_power) / max(len(spec_band) - 1, 1)
        if rest_power <= 0:
            quality = 0.0
        else:
            ratio = peak_power / rest_power
            QMAX = 5.0
            quality = min(ratio, QMAX) / QMAX

        # Update cache + timestamp
        self.last_hr = hr
        self.last_quality = quality
        self.last_run_time = now

        return hr, quality
