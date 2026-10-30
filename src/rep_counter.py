import time

# Index positions inside the 8-element angle vector from get_angle_vector():
# [l_elbow, r_elbow, l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee]
ANGLE_INDEX = {
    "elbow":    (0, 1),
    "shoulder": (2, 3),
    "hip":      (4, 5),
    "knee":     (6, 7),
}

# Config keyed EXACTLY by your raw_videos folder names (case/spacing matters —
# these become your class labels after training)
EXERCISE_CONFIG = {
    "bicep_curl":   {"mode": "reps",  "angle": "elbow", "down": 50,  "up": 150},
    "leg raises":   {"mode": "reps",  "angle": "hip",   "down": 100, "up": 165},
    "plank":        {"mode": "timer", "angle": "hip",   "hold_min": 160, "hold_max": 190},
    "pull Up":      {"mode": "reps",  "angle": "elbow", "down": 50,  "up": 160},
    "push-up":      {"mode": "reps",  "angle": "elbow", "down": 90,  "up": 160},
    "squat":        {"mode": "reps",  "angle": "knee",  "down": 100, "up": 160},
    "tricep dips":  {"mode": "reps",  "angle": "elbow", "down": 90,  "up": 160},
}


class RepCounter:
    def __init__(
        self,
        grace_period_sec=1.5,
        low_confidence_reset_sec=2.0,
        minimum_rep_interval_sec=0.8,
    ):
        self.stage = "up"
        self.count = 0
        self.current_exercise = None
        self.minimum_rep_interval_sec = minimum_rep_interval_sec
        self.last_rep_time = None

        # timer-mode state (for plank, etc.)
        self.timer_active = False
        self.timer_start = None
        self.timer_elapsed = 0.0
        self.last_good_form_time = None
        self.grace_period_sec = grace_period_sec  # allow brief form breaks without resetting

        # Only restart the rep cycle after a sustained low-confidence period.
        self.low_confidence_reset_sec = low_confidence_reset_sec
        self.low_confidence_since = None
        self.last_valid_confidence = None

    def _get_tracked_angle(self, angle_name, angles):
        l_idx, r_idx = ANGLE_INDEX[angle_name]
        return (angles[l_idx] + angles[r_idx]) / 2

    def _reset_for_new_exercise(self, exercise):
        self.current_exercise = exercise
        self.stage = "up"
        self.count = 0
        self.last_rep_time = None
        self.timer_active = False
        self.timer_start = None
        self.timer_elapsed = 0.0
        self.last_good_form_time = None
        self.low_confidence_since = None
        self.last_valid_confidence = None
        print(f"Switched to: {exercise} | Stage: {self.stage} | Count: {self.count}")

    def _handle_confidence_gate(self, exercise, confidence, now):
        if confidence is None:
            return False

        if confidence >= 0.5:
            self.low_confidence_since = None
            self.last_valid_confidence = confidence
            return False

        if self.low_confidence_since is None:
            self.low_confidence_since = now
            return True

        if now - self.low_confidence_since >= self.low_confidence_reset_sec:
            self.stage = "up"
            self.count = 0
            self.last_rep_time = None
            self.low_confidence_since = now
            print(f"Reset rep cycle for {exercise} after sustained low confidence")
            return True

        return True

    def update(self, exercise, angles, confidence=None, movement_valid=True):
        """
        exercise: string label from Module 4
        angles: 8-element angle vector from get_angle_vector()
        confidence: optional model confidence from 0 to 1; low confidence is used
                    to freeze and eventually reset the rep cycle intentionally.
        Returns: dict with either {'mode': 'reps', 'count': N}
                          or       {'mode': 'timer', 'seconds': N}
        """
        now = time.time()

        if exercise in (None, "unknown", "collecting..."):
            if self.current_exercise is not None and confidence is not None:
                self._handle_confidence_gate(self.current_exercise, confidence, now)
            return {"mode": "reps" if self.current_exercise in EXERCISE_CONFIG and EXERCISE_CONFIG[self.current_exercise]["mode"] == "reps" else "timer" if self.current_exercise in EXERCISE_CONFIG and EXERCISE_CONFIG[self.current_exercise]["mode"] == "timer" else "unknown",
                    "value": self.count if self.current_exercise is not None else 0}

        config = EXERCISE_CONFIG.get(exercise)
        if config is None:
            return {"mode": "unknown", "value": self.count if self.current_exercise is not None else 0}

        if exercise != self.current_exercise:
            self._reset_for_new_exercise(exercise)

        if confidence is not None and confidence < 0.5:
            self._handle_confidence_gate(exercise, confidence, now)
            return {"mode": "reps" if config["mode"] == "reps" else "timer",
                    "value": self.count if self.current_exercise is not None else 0}

        tracked_value = self._get_tracked_angle(config["angle"], angles)

        # ---------------- REP-COUNTING MODE ----------------
        if config["mode"] == "reps":
            if not movement_valid:
                self.stage = "up"
                return {"mode": "reps", "value": self.count}

            if tracked_value < config["down"]:
                self.stage = "down"
            if tracked_value > config["up"] and self.stage == "down":
                self.stage = "up"
                if (
                    self.last_rep_time is None
                    or now - self.last_rep_time >= self.minimum_rep_interval_sec
                ):
                    self.count += 1
                    self.last_rep_time = now
            return {"mode": "reps", "value": self.count}

        # ---------------- TIMER MODE (e.g. plank) ----------------
        elif config["mode"] == "timer":
            in_position = (
                movement_valid
                and config["hold_min"] <= tracked_value <= config["hold_max"]
            )
            now = time.time()

            if in_position:
                if not self.timer_active:
                    self.timer_active = True
                    self.timer_start = now
                self.last_good_form_time = now
            else:
                # allow a short grace period (e.g. brief wobble) before pausing the timer
                if self.timer_active and self.last_good_form_time:
                    if now - self.last_good_form_time > self.grace_period_sec:
                        self.timer_elapsed += now - self.timer_start
                        self.timer_active = False
                        self.timer_start = None

            current_session = (now - self.timer_start) if self.timer_active else 0.0
            total_seconds = self.timer_elapsed + current_session
            return {"mode": "timer", "value": round(total_seconds, 1)}

        return {"mode": "unknown", "value": 0}