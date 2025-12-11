# eval_ubfc_video.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import mediapipe as mp

from rppg_estimator import RPPGEstimator
from blink_estimator import BlinkEstimator
from stress_fusion import StressFusion
from utils import eye_aspect_ratio

# ----------------------------------------------------------------------
# CONFIG: set these paths to your UBFC-Phys subject
# ----------------------------------------------------------------------
# Example:
#   subject_01/
#       T1_rest.avi
#       T2_speech.avi
#       T3_arithmetic.avi
#       signals.txt or EDA.txt / BVP.txt / <whatever the authors used>
#
# You WILL need to open the signals file once in a text editor / notebook
# to see column order and sampling rate, then adapt the loader below.

VIDEO_PATH = r"path/to/your/subject/T1_rest.avi"   # Relax video
GT_SIGNAL_PATH = r"path/to/your/subject/EDA.txt"   # or BVP/HR file
GT_IS_EDA = True            # True = use EDA; False = use HR if present

EDA_FS = 4.0                # Empatica E4 EDA is often 4 Hz; adjust if needed
HR_FS = 1.0                 # If you have 1 Hz HR samples; adjust if needed

# ----------------------------------------------------------------------
# Ground truth loader (YOU MUST ADAPT TO YOUR FILE FORMAT)
# ----------------------------------------------------------------------

def load_ubfc_signal(path, is_eda=True):
    """
    Very generic loader:
    - Assumes a text-like file with either:
        time, HR, EDA      or
        just EDA values    or
        just HR values
    - You MUST inspect your file and adapt:
        - which column to use
        - sampling frequency

    Returns:
        t_gt: np.array of times in seconds
        y_gt: np.array of EDA or HR
    """
    data = np.loadtxt(path)

    # EXAMPLE 1: file has only one column (signal only)
    if data.ndim == 1:
        y = data
        fs = EDA_FS if is_eda else HR_FS
        t = np.arange(len(y)) / fs
        return t, y

    # EXAMPLE 2: file has 3 columns: time, HR, EDA
    # uncomment and adapt if that's your case:
    # t = data[:, 0]
    # hr = data[:, 1]
    # eda = data[:, 2]
    # return t, (eda if is_eda else hr)

    # default fallback: treat 1st column as time, 2nd as signal
    t = data[:, 0]
    y = data[:, 1]
    return t, y


# ----------------------------------------------------------------------
# Run your pipeline on a video (offline, no webcam window)
# ----------------------------------------------------------------------

def run_pipeline_on_video(video_path):
    """
    Runs the stress pipeline on a UBFC-Phys video.
    Returns:
        t_est: time stamps (s from start of video)
        stress_scores: StressIndex(t) in [0,100]
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # same objects as webcam script
    rppg = RPPGEstimator(window_seconds=20.0)
    blink = BlinkEstimator(window_seconds=20.0)
    fusion = StressFusion()

    # mediapipe eye indices
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

    t_list = []
    stress_list = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_curr = frame_idx / fps
        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        results = face_mesh.process(frame_rgb)

        hr, q = None, 0.0
        blink_rate = 0.0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # --- rPPG update
            rppg.update(frame, face_landmarks)
            hr, q = rppg.estimate_hr_and_quality()

            # --- blink via EAR
            def get_pts(indices):
                return np.array(
                    [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in indices]
                )

            left_eye_pts = get_pts(LEFT_EYE_IDX)
            right_eye_pts = get_pts(RIGHT_EYE_IDX)

            ear_left = eye_aspect_ratio(left_eye_pts)
            ear_right = eye_aspect_ratio(right_eye_pts)
            ear = 0.5 * (ear_left + ear_right)

            # use video time as "current_time" so blink rate is consistent
            blink.update(ear, current_time=t_curr)
            blink_rate = blink.blink_rate_bpm()

        stress_score, S_phy, S_beh = fusion.update(hr, q, blink_rate)

        t_list.append(t_curr)
        stress_list.append(stress_score)

    cap.release()
    cv2.destroyAllWindows()

    return np.array(t_list), np.array(stress_list)


# ----------------------------------------------------------------------
# Utility: resample one signal to another time axis
# ----------------------------------------------------------------------

def resample_to(t_src, y_src, t_target):
    if len(t_src) < 2:
        return np.full_like(t_target, np.nan, dtype=float)
    return np.interp(t_target, t_src, y_src)


# ----------------------------------------------------------------------
# Main: run + plot
# ----------------------------------------------------------------------

def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not os.path.exists(GT_SIGNAL_PATH):
        raise FileNotFoundError(f"GT signal not found: {GT_SIGNAL_PATH}")

    print("Running pipeline on UBFC-Phys video...")
    t_est, stress_scores = run_pipeline_on_video(VIDEO_PATH)

    print("Loading ground-truth signal...")
    t_gt, y_gt = load_ubfc_signal(GT_SIGNAL_PATH, is_eda=GT_IS_EDA)

    # Align time range
    t_min = max(t_est[0], t_gt[0])
    t_max = min(t_est[-1], t_gt[-1])
    if t_max <= t_min:
        raise ValueError("No overlap between video time and GT time; check your files.")

    # Common time grid for plotting
    t_common = np.linspace(t_min, t_max, 300)

    stress_common = resample_to(t_est, stress_scores, t_common)
    gt_common = resample_to(t_gt, y_gt, t_common)

    # Normalize GT if you like (for visual comparison only)
    gt_norm = (gt_common - np.nanmin(gt_common)) / (np.nanmax(gt_common) - np.nanmin(gt_common) + 1e-6)
    gt_norm *= 100.0  # map to ~0–100 range

    # Plot
    label_gt = "EDA" if GT_IS_EDA else "HR"

    plt.figure(figsize=(10, 5))
    plt.title("UBFC-Phys: Stress_Index vs Ground Truth")
    plt.plot(t_common, stress_common, label="Stress_Index (0–100)", linewidth=2)
    plt.plot(t_common, gt_norm, label=f"{label_gt} (normalized to 0–100)", linewidth=2, alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Value (0–100)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Simple correlation (optional)
    mask = ~np.isnan(stress_common) & ~np.isnan(gt_common)
    if np.sum(mask) > 10:
        r = np.corrcoef(stress_common[mask], gt_common[mask])[0, 1]
        print(f"Correlation Stress_Index vs {label_gt}: {r:.3f}")
    else:
        print("Not enough valid points to compute correlation.")


if __name__ == "__main__":
    main()
