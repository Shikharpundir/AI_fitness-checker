import numpy as np
import pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model


class DLExerciseClassifier:
    def __init__(self, model_path="models/exercise_lstm.h5",
                 encoder_path="models/label_encoder.pkl", seq_len=30,
                 confidence_threshold=0.6,
                 confidence_margin=0.15,
                 smoothing_window=10,
                 predict_every_n=3,
                 switch_streak=5):
        self.model = load_model(model_path)
        with open(encoder_path, "rb") as f:
            self.encoder = pickle.load(f)

        self.buffer = deque(maxlen=seq_len)
        self.seq_len = seq_len
        self.confidence_threshold = confidence_threshold
        self.confidence_margin = confidence_margin

        self.recent_predictions = deque(maxlen=smoothing_window)
        self.predict_every_n = predict_every_n
        self.frame_counter = 0
        self.switch_streak = switch_streak
        self.candidate_label = None
        self.candidate_count = 0

        self.stable_label = "collecting..."
        self.stable_confidence = 0.0

    def _raw_predict(self):
        seq = np.expand_dims(np.array(self.buffer), axis=0)
        probs = self.model.predict(seq, verbose=0)[0]

        sorted_idx = np.argsort(probs)[::-1]
        top_idx, second_idx = sorted_idx[:2]
        top_prob, second_prob = probs[top_idx], probs[second_idx]
        margin = top_prob - second_prob

        if top_prob >= self.confidence_threshold and margin >= self.confidence_margin:
            label = self.encoder.inverse_transform([top_idx])[0]
        else:
            label = "unknown"

        return label, float(top_prob)

    def update(self, angle_vector):
        # Keep inference features on the same [0, 1] scale used during training.
        self.buffer.append(np.asarray(angle_vector, dtype="float32") / 180.0)
        self.frame_counter += 1

        if len(self.buffer) < self.seq_len:
            return "collecting...", 0.0

        if self.frame_counter % self.predict_every_n == 0:
            raw_label, raw_confidence = self._raw_predict()
            self.recent_predictions.append(raw_label)

            # Only allow a label change after the recent predictions agree.
            most_common_label, count = Counter(self.recent_predictions).most_common(1)[0]
            majority_ok = count >= (len(self.recent_predictions) // 2 + 1)

            if most_common_label == self.stable_label:
                self.candidate_label = None
                self.candidate_count = 0
            elif majority_ok:
                if most_common_label == self.candidate_label:
                    self.candidate_count += 1
                else:
                    self.candidate_label = most_common_label
                    self.candidate_count = 1

                if self.candidate_count >= self.switch_streak:
                    self.stable_label = most_common_label
                    self.candidate_label = None
                    self.candidate_count = 0

            if (raw_label == self.stable_label or
                    self.stable_label in ("collecting...", "unknown")):
                self.stable_confidence = raw_confidence

        return self.stable_label, self.stable_confidence