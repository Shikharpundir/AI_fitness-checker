import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pathlib
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_FILENAME = "pose_landmarker_lite.task"


def download_model(model_path: pathlib.Path) -> pathlib.Path:
    if model_path.exists() and model_path.stat().st_size > 500_000:
        print(f"Model already present: {model_path} ({model_path.stat().st_size} bytes)")
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe model to {model_path} ...")
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
        model_path.write_bytes(response.read())
    print(f"Download complete. Size: {model_path.stat().st_size} bytes")
    return model_path


def draw_landmarks_on_frame(frame, detection_result):
    if not detection_result.pose_landmarks:
        return frame
    for pose_landmarks in detection_result.pose_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return frame


def get_landmark_list(detection_result, frame_shape):
    h, w = frame_shape[:2]
    landmarks = []
    if detection_result.pose_landmarks:
        for lm in detection_result.pose_landmarks[0]:
            landmarks.append((lm.x * w, lm.y * h, lm.z, lm.visibility))
    return landmarks


def main() -> None:
    model_path = pathlib.Path(__file__).parent / MODEL_FILENAME
    download_model(model_path)

    print("Loading PoseLandmarker...")
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    print("PoseLandmarker loaded.")

    cap = cv2.VideoCapture(0)
    print("Camera opened:", cap.isOpened())
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera — try index 1, 2, etc.")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed, stopping.")
                break

            frame_count += 1
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                frame = draw_landmarks_on_frame(frame, result)
                landmarks = get_landmark_list(result, frame.shape)

                cv2.putText(frame, f"Landmarks: {len(landmarks)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: {len(landmarks)} landmarks. "
                          f"Left elbow (idx 13): {landmarks[13]}")
            else:
                cv2.putText(frame, "No person detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Pose Detection Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("Cleaned up and exited.")


if __name__ == "__main__":
    main()