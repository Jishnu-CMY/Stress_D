# main_webcam.py
import time

import cv2
import mediapipe as mp
import numpy as np

#from rppg_estimator import RPPGEstimator
from rppg_deepphys import DeepPhysEstimator as RPPGEstimator
from blink_estimator import BlinkEstimator
from stress_fusion import StressFusion
from utils import eye_aspect_ratio


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    # Indices for eyes (MediaPipe FaceMesh reference)
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

    rppg = RPPGEstimator(weights_path="weights/UBFC-rPPG_DeepPhys.pth",
                     window_seconds=20.0,
                     img_size=72,
                     min_interval=0.5)
    
    blink = BlinkEstimator(window_seconds=20.0)
    fusion = StressFusion()

    last_print = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        results = face_mesh.process(frame_rgb)

        hr, q = None, 0.0
        blink_rate = 0.0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # --------------------------
            # Physiological: rPPG update
            # --------------------------
            rppg.update(frame, face_landmarks)
            hr, q = rppg.estimate_hr_and_quality()

            # --------------------------
            # Behavioral: Blink via EAR
            # --------------------------
            def get_pts(indices):
                return np.array(
                    [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in indices]
                )

            left_eye_pts = get_pts(LEFT_EYE_IDX)
            right_eye_pts = get_pts(RIGHT_EYE_IDX)

            ear_left = eye_aspect_ratio(left_eye_pts)
            ear_right = eye_aspect_ratio(right_eye_pts)
            ear = 0.5 * (ear_left + ear_right)

            now = time.time()
            blink.update(ear, now)
            blink_rate = blink.blink_rate_bpm()

            # Draw eyes for visualization
            for pt in np.vstack([left_eye_pts, right_eye_pts]).astype(int):
                cv2.circle(frame, tuple(pt), 1, (0, 255, 0), -1)

        # --------------------------
        # Stress fusion
        # --------------------------
        stress_score, S_phy, S_beh = fusion.update(hr, q, blink_rate)

        # --------------------------
        # Color choice based on stress
        # --------------------------
        if stress_score < 30:
            text_color = (0, 220, 255)    # Calm - Cyan
        elif stress_score < 70:
            text_color = (0, 255, 255)    # Medium - Yellow
        else:
            text_color = (0, 0, 255)      # High Stress - Red

        # --------------------------
        # Overlay info on frame
        # --------------------------
        text_lines = [
            f"Stress: {stress_score:3d} / 100",
            f"HR: {hr:.1f} bpm" if hr is not None else "HR: --",
            f"rPPG quality: {q:.2f}",
            f"Blink rate: {blink_rate:.1f} / min"
        ]

        # Font styling
        font        = cv2.FONT_HERSHEY_TRIPLEX
        font_scale  = 1.0      # bigger than before
        thickness   = 3        # thicker for visibility
        outline_col = (0, 0, 0)  # Black shadow

        y0 = 40
        line_spacing = 40

        for i, line in enumerate(text_lines):
            y = y0 + line_spacing * i
            # Outline (shadow)
            cv2.putText(
                frame, line, (10, y),
                font, font_scale, outline_col, thickness + 2
            )
            # Foreground (stress-based color)
            cv2.putText(
                frame, line, (10, y),
                font, font_scale, text_color, thickness
            )

        cv2.imshow("Multi-Signal Stress Detector", frame)

        # Optional: print to console at ~1 Hz
        now = time.time()
        if now - last_print > 1.0:
            last_print = now
            print(
                f"Stress={stress_score:3d}, HR={hr}, Q={q:.2f}, "
                f"BR={blink_rate:.1f}, S_phy={S_phy:.2f}, S_beh={S_beh:.2f}"
            )

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
