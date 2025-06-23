"""
run_webcam_bicep.py
Press Q to quit the stream.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
#  project imports
# ──────────────────────────────────────────────────────────────
from config import sport_list
from utils.angle import calculate_average_joint_angle
from utils.plot import plot
from utils.visual import put_text


# ──────────────────────────────────────────────────────────────
#  simple, frame-level feedback helper
# ──────────────────────────────────────────────────────────────
def get_form_feedback(angle, maintaining, relaxing, tol=10):
    """
    Very simple heuristic.  Tweak thresholds or add more states later.
    """
    if angle < maintaining - tol:
        return "Straighten arm ↘︎"
    elif angle > relaxing + tol:
        return "Curl higher ↗︎"
    else:
        return "Good form ✓"


# ──────────────────────────────────────────────────────────────
#  main loop
# ──────────────────────────────────────────────────────────────
def run_bicep_curl_cam(cam_id: int = 0):
    exercise = "deadlift"          # key in sport_list
    model = YOLO("yolo11n-pose.pt")  # or yolov8n-pose.pt if that’s your file
    cap   = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam id {cam_id}")

    count, relax_confirm = 0, 0
    prev_state           = "relaxed"

    l_idx  = sport_list[exercise]["left_points_idx"]
    r_idx  = sport_list[exercise]["right_points_idx"]
    keeping= sport_list[exercise]["maintaining"]
    relax  = sport_list[exercise]["relaxing"]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── model inference ───────────────────────────────────
        res = model(frame, verbose=False)[0]
        if res.keypoints.shape[1] == 0:         # no person detected
            cv2.imshow("YOLOv8 Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        angle = calculate_average_joint_angle(res.keypoints, l_idx, r_idx)

        # ── rep counter state-machine ─────────────────────────
        if prev_state == "relaxed" and angle < keeping:
            prev_state = "active"
        elif prev_state == "active" and angle > relax:
            relax_confirm += 1
            if relax_confirm >= 1:              # debounce
                count += 1
                prev_state    = "relaxed"
                relax_confirm = 0

        # ── visualisation ─────────────────────────────────────
        scale = max(frame.shape[1]/960, frame.shape[0]/540)
        vis   = plot(res, scale)                      # skeleton
        feedback = get_form_feedback(angle, keeping, relax)
        put_text(vis, "Bicep Curl", count, 0, scale)  # existing HUD
        cv2.putText(                                   # new feedback line
            vis, f"Form: {feedback}", (int(10*scale), int(120*scale)),
            0, 0.7*scale,
            (0,255,0) if "Good" in feedback else (0,0,255),
            thickness=int(2*scale), lineType=cv2.LINE_AA
        )

        cv2.imshow("YOLOv8 Pose", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────
#  entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_bicep_curl_cam(0)   # change cam_id if you have multiple cameras