# stress_fusion.py

# ============================================================
# Stress Fusion
# ============================================================
from utils import lowpass_filter
import numpy as np
import time
from collections import deque


class StressFusion:
    def __init__(self, lowpass_cutoff=1.5, lowpass_order=3, ema_alpha=0.85):
        # Hyperparameters
        self.HR_MIN = 55.0
        self.HR_MAX = 120.0
        self.BR_LOW = 10.0
        self.BR_HIGH = 40.0

        self.w_phy = 0.6
        self.w_beh = 0.4
        self.alpha = 0.8  # temporal smoothing

        self.prev_S = 0.0

        self.lowpass_cutoff = lowpass_cutoff  # Hz, e.g., 1.5
        self.lowpass_order = lowpass_order
        self.prev_S = 0.0
        self.prev_time = None
        # buffers for recent HR/blink samples to perform lowpass
        self.hr_buffer = deque()
        self.hr_time = deque()
        self.br_buffer = deque()
        self.br_time = deque()


    def physiological_index(self, hr, quality):
        if hr is None or quality <= 0:
            return 0.0

        s_hr = (hr - self.HR_MIN) / (self.HR_MAX - self.HR_MIN)
        s_hr = max(0.0, min(1.0, s_hr))

        return s_hr * quality  # 0–1

    def behavioral_index(self, blink_rate):
        s_b = (blink_rate - self.BR_LOW) / (self.BR_HIGH - self.BR_LOW)
        s_b = max(0.0, min(1.0, s_b))
        return s_b

    def _push_sample(self, buffer, times, value, tnow, max_seconds=10.0):
        buffer.append(value)
        times.append(tnow)
        # trim to max_seconds
        while len(times) > 2 and (times[-1] - times[0] > max_seconds):
            buffer.popleft(); times.popleft()

    def _lowpass_from_buffer(self, buffer, times, cutoff):
        if len(buffer) < 3:
            return None
        arr = np.array(buffer)
        # infer fs from timestamps
        duration = times[-1] - times[0]
        if duration <= 0:
            return None
        fs = len(arr) / duration
        try:
            out = lowpass_filter(arr, cutoff=cutoff, fs=fs, order=self.lowpass_order)
        except Exception:
            return None
        # return last value (most recent smoothed)
        return float(out[-1])

    def update(self, hr, quality, blink_rate):
        tnow = time.time()
        # push samples to buffers (use NaN-safe)
        if hr is not None:
            self._push_sample(self.hr_buffer, self.hr_time, float(hr), tnow)
        if blink_rate is not None:
            self._push_sample(self.br_buffer, self.br_time, float(blink_rate), tnow)

        # compute lowpass-smoothed hr and br
        hr_smooth = self._lowpass_from_buffer(self.hr_buffer, self.hr_time, self.lowpass_cutoff)
        br_smooth = self._lowpass_from_buffer(self.br_buffer, self.br_time, self.lowpass_cutoff)

        # fallback to raw if smoothing not available
        hr_used = hr_smooth if hr_smooth is not None else hr
        br_used = br_smooth if br_smooth is not None else blink_rate

        # proceed with existing physiological_index / behavioral_index using hr_used/br_used
        S_phy = self.physiological_index(hr_used, quality)
        S_beh = self.behavioral_index(br_used)

        # weight physiology by quality
        w_phy_eff = self.w_phy * quality
        w_beh_eff = self.w_beh

        denom = w_phy_eff + w_beh_eff
        S_raw = (w_phy_eff * S_phy + w_beh_eff * S_beh) / denom if denom > 0 else 0.0

        # temporal smoothing via EMA (keep existing alpha)
        S = self.alpha * self.prev_S + (1.0 - self.alpha) * S_raw
        self.prev_S = S
        stress_score = int(round(100.0 * S))
        stress_score = max(0, min(100, stress_score))
        return stress_score, S_phy, S_beh
