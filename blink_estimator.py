# blink_estimator.py
import time
from collections import deque
import numpy as np

class BlinkEstimator:
    def __init__(self,
                 ear_threshold=0.23,
                 min_closed_frames=2,
                 max_closed_frames=8,
                 window_seconds=30.0,
                 ema_alpha=0.18):
        """
        ema_alpha: smoothing factor for blink rate EMA (0-1). Larger => more responsive.
        window_seconds: keep blink events for rate calculation.
        """
        self.ear_threshold = ear_threshold
        self.min_closed_frames = min_closed_frames
        self.max_closed_frames = max_closed_frames
        self.window_seconds = window_seconds

        self.closed_frames = 0
        self.prev_closed = False
        self.blink_times = deque()

        # EMA of instantaneous blink rate (blinks/min)
        self.ema_alpha = ema_alpha
        self.smoothed_blink_rate = 0.0

    def update(self, ear, current_time=None):
        if current_time is None:
            current_time = time.time()

        is_closed = ear < self.ear_threshold

        if is_closed:
            self.closed_frames += 1
        else:
            # eye opened — check if it was a valid blink
            if self.prev_closed:
                if self.min_closed_frames <= self.closed_frames <= self.max_closed_frames:
                    self.blink_times.append(current_time)
            self.closed_frames = 0

        self.prev_closed = is_closed

        # Trim old blink events
        while len(self.blink_times) and (current_time - self.blink_times[0] > self.window_seconds):
            self.blink_times.popleft()

        # Update EMA of blink rate every update step using instantaneous estimate
        inst_rate = self._instant_blink_rate(current_time)
        # If inst_rate is NaN/0, still apply EMA so value decays smoothly
        self.smoothed_blink_rate = (self.ema_alpha * inst_rate +
                                    (1.0 - self.ema_alpha) * self.smoothed_blink_rate)

    def _instant_blink_rate(self, current_time):
        # compute blink rate in blinks/min over the recent window
        if len(self.blink_times) < 2:
            return 0.0
        dt = self.blink_times[-1] - self.blink_times[0]
        if dt <= 0:
            return 0.0
        blinks = len(self.blink_times)
        return (blinks / dt) * 60.0

    def blink_rate_bpm(self):
        """Return EMA-smoothed blink rate (blinks per minute)."""
        return float(self.smoothed_blink_rate)
