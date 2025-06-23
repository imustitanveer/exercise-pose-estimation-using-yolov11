"""
main.py
Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
"""

import base64, io, time, json
from typing import Dict

import cv2, numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

from config import sport_list
from utils.angle import calculate_angle_cosine     # <- direct fn import
from utils.plot import plot
from utils.visual import put_text

# ─────────────────────────────────────────────────────────────
#  tiny helper: convert NumPy scalars to native python for json
# ─────────────────────────────────────────────────────────────
def pyify(o):
    if isinstance(o, (np.floating,)):   # float32/64
        return float(o)
    if isinstance(o, (np.integer,)):    # int32/64
        return int(o)
    if isinstance(o, np.ndarray):       # any array → list
        return o.tolist()
    raise TypeError(f"{o!r} is not JSON serialisable")

# ── initialise FastAPI & model
app   = FastAPI(title="Pose-rep API")
model = YOLO("yolo11n-pose.pt")         # load once at startup


# ─────────────────────────────────────────────────────────────
#  Helper: process ONE frame
# ─────────────────────────────────────────────────────────────
def process_frame(img_bgr: np.ndarray,
                  exercise: str,
                  session: Dict) -> Dict:
    t0 = time.time()
    res = model(img_bgr, verbose=False)[0]
    if res.keypoints.shape[1] == 0:
        return {"skip": True}

    cfg   = sport_list[exercise]
    lidx  = cfg["left_points_idx"]
    ridx  = cfg["right_points_idx"]
    keep  = cfg["maintaining"]
    relax = cfg["relaxing"]

    kpts = res.keypoints.data[0]
    la   = calculate_angle_cosine(kpts[lidx[0]], kpts[lidx[1]], kpts[lidx[2]])
    ra   = calculate_angle_cosine(kpts[ridx[0]], kpts[ridx[1]], kpts[ridx[2]])
    angle = float(min(la, ra))          # cast → python float

    # ── rep-counter (3-frame debounce)
    if session["state"] == "relaxed" and angle < keep:
        session["state"] = "active"
    elif session["state"] == "active" and angle > relax:
        session["debounce"] += 1
        if session["debounce"] >= 3:
            session["count"] += 1
            session["state"] = "relaxed"
            session["debounce"] = 0
    else:
        session["debounce"] = 0

    feedback = (
        "Straighten arm ↘︎" if angle < keep - 10 else
        "Curl higher ↗︎"   if angle > relax + 10 else
        "Good form ✓"
    )

    scale = max(img_bgr.shape[1] / 960, img_bgr.shape[0] / 540)
    vis   = plot(res, scale)
    put_text(vis, exercise, session["count"], 0, scale)
    cv2.putText(
        vis, f"Form: {feedback}", (int(10*scale), int(120*scale)),
        0, 0.7*scale,
        (0, 255, 0) if "Good" in feedback else (0, 0, 255),
        thickness=int(2*scale), lineType=cv2.LINE_AA
    )

    _, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    jpeg_b64 = base64.b64encode(buf).decode()

    fps = float(round(1.0 / max(1e-6, time.time() - t0), 1))

    return {
        "frame": jpeg_b64,
        "count": int(session["count"]),
        "angle": angle,
        "feedback": feedback,
        "fps": fps
    }


# ─────────────────────────────────────────────────────────────
#  WebSocket endpoint
# ─────────────────────────────────────────────────────────────
@app.websocket("/ws/{exercise}")
async def websocket_endpoint(ws: WebSocket, exercise: str):
    if exercise not in sport_list:
        await ws.close(code=4001)
        return

    await ws.accept()
    session = {"count": 0, "state": "relaxed", "debounce": 0}

    try:
        while True:
            msg = await ws.receive_json()
            img_b64 = msg.get("frame")
            if img_b64 is None:
                continue
            img_np = np.frombuffer(base64.b64decode(img_b64), np.uint8)
            frame  = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            result = process_frame(frame, exercise, session)
            if result.get("skip"):
                continue

            # ******* key line: use json.dumps with default=pyify *******
            await ws.send_text(json.dumps(result, default=pyify))

    except WebSocketDisconnect:
        print("client disconnected")


# ─────────────────────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    return {"status": "ok"}