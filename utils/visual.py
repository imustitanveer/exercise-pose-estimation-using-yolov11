import cv2

def put_text(frame, exercise, count, fps, redio):
    cv2.putText(frame, f'Exercise: {exercise}', (int(10 * redio), int(30 * redio)), 0, 0.7 * redio,
                (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA)
    cv2.putText(frame, f'Count: {count}', (int(10 * redio), int(60 * redio)), 0, 0.7 * redio,
                (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA)
    cv2.putText(frame, f'FPS: {fps}', (int(10 * redio), int(90 * redio)), 0, 0.7 * redio,
                (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA)