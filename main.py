import cv2
from ultralytics import YOLO
from config import sport_list
from utils.angle import calculate_average_joint_angle
from utils.plot import plot
from utils.visual import put_text

def main(value, name):
    model = YOLO('yolo11n-pose.pt')
    cap = cv2.VideoCapture(name)
    count = 0
    prev_state = "relaxed"
    relax_confirm = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        plot_size_ratio = max(frame.shape[1] / 960, frame.shape[0] / 540)
        results = model(frame, verbose=False)

        if results[0].keypoints.shape[1] == 0:
            continue

        left_idx = sport_list[value]['left_points_idx']
        right_idx = sport_list[value]['right_points_idx']
        maintaining = sport_list[value]['maintaining']
        relaxing = sport_list[value]['relaxing']

        angle = calculate_average_joint_angle(results[0].keypoints, left_idx, right_idx)

        if prev_state == "relaxed" and angle < maintaining:
            prev_state = "active"
        elif prev_state == "active" and angle > relaxing:
            relax_confirm += 1
            if relax_confirm >= 1:
                count += 1
                prev_state = "relaxed"
                relax_confirm = 0

        annotated_frame = plot(results[0], plot_size_ratio)
        put_text(annotated_frame, value, count, 0, plot_size_ratio)
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(value="deadlift", name="deadlift.MOV")
#    main(value="squat", name="squats.mp4")
#    main(value="bench", name="bench.MOV")
#    main(value="bench", name="bench2.mp4")
    