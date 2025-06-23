import asyncio, base64, json, sys, time, cv2, numpy as np, websockets

URI     = sys.argv[1] if len(sys.argv) >= 2 else "ws://127.0.0.1:8000/ws/bicep_curl"
CAM_ID  = int(sys.argv[2]) if len(sys.argv) >= 3 else 0         # built-in cam =0, phone cam =1/2
JPEG_Q  = 80

def encode_jpeg(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf).decode()

def decode_jpeg(b64):
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

async def main():
    # ── open camera first; abort early if it fails
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera id {CAM_ID}")

    print(f"▶ Connecting to {URI} …")
    async with websockets.connect(URI, max_size=2**23) as ws:   # 8 MB cap
        print("✅ WebSocket open")
        fps_clock = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed — exiting")
                break

            # ── send frame up
            await ws.send(json.dumps({"frame": encode_jpeg(frame)}))

            # ── wait for result
            msg = await ws.recv()          # server always replies
            data = json.loads(msg)

            # ── overlay & show
            vis  = decode_jpeg(data["frame"])
            cv2.imshow("Annotated stream", vis)

            # simple FPS monitor for client ↔ server rtt
            if int(time.time() - fps_clock) >= 1:
                print(f"reps: {data['count']}, angle: {data['angle']:.1f}, feedback: {data['feedback']}")
                fps_clock = time.time()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("🔚 closed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print("❌", e)