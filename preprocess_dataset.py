import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tqdm import tqdm

import pathlib
import numpy as np
import cv2

from pose_test import PoseDetector
from angle_utils import get_angle_vector

RAW_DIR = pathlib.Path("data/raw_videos")
OUT_DIR = pathlib.Path("data/processed_sequences")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 30  # fixed number of frames per training sample

detector = PoseDetector()
X, y = [], []

for label_dir in sorted(RAW_DIR.iterdir()):
    if not label_dir.is_dir():
        continue
    label = label_dir.name
    video_files = list(label_dir.glob("*.mp4")) + list(label_dir.glob("*.avi")) + list(label_dir.glob("*.mov"))
    print(f"Processing '{label}': {len(video_files)} clips")

    for video_path in tqdm(video_files, desc=f"  {label}"):
        cap = cv2.VideoCapture(str(video_path))
        seq = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame, result = detector.find_pose(frame, draw=False)
            landmarks = detector.get_landmark_list(result, frame.shape)
            if landmarks:
                seq.append(get_angle_vector(landmarks))
        cap.release()

        if len(seq) < 5:  # skip clips where pose barely detected
            print(f"  Skipping {video_path.name} (too few valid frames: {len(seq)})")
            continue

        # resample to exactly SEQ_LEN frames (evenly spaced)
        idxs = np.linspace(0, len(seq) - 1, SEQ_LEN).astype(int)
        seq_resampled = np.array(seq)[idxs]

        X.append(seq_resampled)
        y.append(label)

X = np.array(X)   # shape: (num_samples, 30, 8)
y = np.array(y)   # shape: (num_samples,)

np.save(OUT_DIR / "X.npy", X)
np.save(OUT_DIR / "y.npy", y)

print(f"\nDone. X shape: {X.shape}, y shape: {y.shape}")
print("Class distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u}: {c}")

detector.close()