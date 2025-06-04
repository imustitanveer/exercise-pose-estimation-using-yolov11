# 🏋️‍♂️ PoseRep: Smart Exercise Rep Counter with YOLOv11

PoseRep is a real-time pose estimation and rep counting tool powered by [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics). It analyzes body keypoints during exercise to count reps, track form, and visualize motion — making it perfect for building smart gym assistants or fitness apps.  

Currently supports:
- 🏋️‍♀️ Squat
- 💪 Bench Press
- 🏋️‍♂️ Deadlift

---

## 🚀 Features

✅ Real-time keypoint detection with YOLOv11  
✅ Angle calculation via cosine similarity  
✅ Intelligent rep counter using joint angle thresholds  
✅ OpenCV visualization with annotated pose skeleton  
✅ Easy config for adding new exercises  

---

## Demo
> Just run the script to see it live!

---

## 🧠 How It Works

- Detects body joints using YOLOv11 pose estimation.
- Calculates angles between shoulder–elbow–wrist or hip–knee–ankle (depending on the exercise).
- Tracks whether a rep is in a "maintaining" or "relaxed" phase based on angle thresholds.
- Counts reps when a full cycle is completed.

---

## 📂 Project Structure

```bash
pose-rep/
├── core/
│   ├── angle_utils.py         # Angle calculations
│   ├── config.py              # Exercise-specific metadata
│   ├── plot_utils.py          # Keypoint + skeleton plotting
├── main.py                    # Entry point for video inference
├── requirements.txt           # Python dependencies
└── README.md
```

---

🛠️ Installation

1.	Clone the repo

```bash
git clone https://github.com/imustitanveer/exercise-pose-estimation-using-yolov11.git
cd pose-rep
```

2.	Install dependencies

```bash
pip install -r requirements.txt
```

3.	Download YOLOv11 pose model

You’ll need yolo11n-pose.pt or similar. Download from Ultralytics releases.

4.	Run it

```bahs
python main.py
```

---

## ✨ Future Plans

🔥 FastAPI backend
Serve inference results via an API for mobile/web clients.

🔥 Uvicorn deployment
Run it as a blazing-fast async server.

🔥 Add more exercises
Support rows, curls, lunges, shoulder press, etc.

🔥 Angle correction feedback
Warn users if form deviates from optimal joint angles.

🔥 Save & export sessions
Track your reps, sets, and form history across workouts.

---

## 🤝 Contributing

Wanna help make this the next smart gym revolution?
PRs and ideas are welcome — just open an issue or fork and improve!

---

📜 License

MIT License. Do whatever, just don’t claim you invented squats.

---

🧠 Author

Built with sweat & swearing by @imustitanveer

💬 DM for collaborations or feature requests!