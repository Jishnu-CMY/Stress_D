# process_video.py
import time
import cv2
import mediapipe as mp
import numpy as np

from rppg_estimator import RPPGEstimator
from blink_estimator import BlinkEstimator
from stress_fusion import StressFusion
from utils import eye_aspect_ratio


def run_video(video_path, output_csv="stress_output3.csv"):
    print(f"Processing: {video_path}")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)

    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

    rppg = RPPGEstimator(
        window_frames=128,
        target_fps=30.0,
        weights_path="weights/PhysNet_pretrained.pth"
    )
    blink = BlinkEstimator(window_seconds=20.0)
    fusion = StressFusion()

    # For saving stress over time
    results = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        results_mp = face_mesh.process(frame_rgb)

        hr, q = None, 0.0
        blink_rate = 0.0

        if results_mp.multi_face_landmarks:
            face_landmarks = results_mp.multi_face_landmarks[0].landmark

            rppg.update(frame, face_landmarks)
            hr, q = rppg.estimate_hr_and_quality()

            def get_pts(indices):
                return np.array([[face_landmarks[i].x * w,
                                  face_landmarks[i].y * h] for i in indices])

            left_eye_pts = get_pts(LEFT_EYE_IDX)
            right_eye_pts = get_pts(RIGHT_EYE_IDX)

            ear_left = eye_aspect_ratio(left_eye_pts)
            ear_right = eye_aspect_ratio(right_eye_pts)
            ear = 0.5 * (ear_left + ear_right)

            blink.update(ear, time.time())
            blink_rate = blink.blink_rate_bpm()

        stress_score, S_phy, S_beh = fusion.update(hr, q, blink_rate)

        # Save results to list
        timestamp = frame_id / 30.0  # UBFC videos are 30 FPS
        results.append([timestamp, stress_score, hr, q, blink_rate])

        print(f"Frame {frame_id}: Stress={stress_score}, HR={hr}, Q={q:.2f}")

    cap.release()

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(results, columns=["Time(s)", "Stress_Index", "HR", "Quality", "Blink_Rate"])
    df.to_csv(output_csv, index=False)
    print("Saved:", output_csv)


if __name__ == "__main__":
    run_video("./S1/vid_s1_T1.avi", "S1_T1_stress.csv")
