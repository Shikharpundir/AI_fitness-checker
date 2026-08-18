import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence background logs

import pathlib
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Configuration Constants
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = pathlib.Path(__file__).parent / "pose_landmarker_lite.task"


def setup_model() -> str:
    """Downloads the MediaPipe model file if it does not exist."""
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 500_000:
        print(f"Downloading model to {MODEL_PATH}...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            MODEL_PATH.write_bytes(response.read())
    return str(MODEL_PATH)


def draw_landmarks(frame, result):
    """Draws skeleton connections on the image frame if a person is detected."""
    if not result.pose_landmarks:
        return
    for pose_landmarks in result.pose_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            frame, proto, mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )


def main():
    # 1. Initialize Pose Detector
    model_file = setup_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_file),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # 2. Open Camera Stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Running... Press 'q' to quit.")
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # MediaPipe expects RGB images, OpenCV defaults to BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # VIDEO mode requires a unique timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # 3. Process and display results
            if result.pose_landmarks:
                draw_landmarks(frame, result)
                
                # Convert normalized landmarks to absolute pixel coordinates
                h, w, _ = frame.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in result.pose_landmarks[0]]
                
                cv2.putText(frame, f"Landmarks detected: {len(landmarks)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No person detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Simplified Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # 4. Resource cleanup
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
