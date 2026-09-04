import numpy as np
import pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model


class DLExerciseClassifier:
    def __init__(self, model_path="models/exercise_lstm.h5",
                 encoder_path="models/label_encoder.pkl", seq_len=30,
                 confidence_threshold=0.35,
                 smoothing_window=8,
                 predict_every_n=3):
        self.model = load_model(model_path)
        with open(encoder_path, "rb") as f:
            self.encoder = pickle.load(f)

        self.buffer = deque(maxlen=seq_len)
        self.seq_len = seq_len
        self.confidence_threshold = confidence_threshold

        self.recent_predictions = deque(maxlen=smoothing_window)
        self.predict_every_n = predict_every_n
        self.frame_counter = 0

        self.stable_label = "collecting..."
        self.stable_confidence = 0.0
        self.last_valid_label = None

    def update(self, angle_vector):
        # Keep inference features on the same [0, 1] scale used during training.
        self.buffer.append(np.asarray(angle_vector, dtype="float32") / 180.0)
        self.frame_counter += 1

        if len(self.buffer) < self.seq_len:
            return "collecting...", 0.0

        if self.frame_counter % self.predict_every_n == 0:
            seq = np.expand_dims(np.array(self.buffer), axis=0)
            probs = self.model.predict(seq, verbose=0)[0]
            best_idx = np.argmax(probs)
            confidence = float(probs[best_idx])

            if confidence >= self.confidence_threshold:
                label = self.encoder.inverse_transform([best_idx])[0]
                self.last_valid_label = label
                self.stable_label = label
                self.stable_confidence = confidence
            else:
                if self.last_valid_label is not None:
                    return self.last_valid_label, self.stable_confidence
                self.stable_label = "unknown"
                self.stable_confidence = confidence

        return self.stable_label, self.stable_confidence