import time


EXERCISE_METS = {
    "bicep_curl": 4.5,
    "leg raises": 4.0,
    "plank": 3.8,
    "pull Up": 6.0,
    "push-up": 8.0,
    "squat": 6.0,
    "tricep dips": 5.0,
}

# Rough estimate of how long a rep takes in seconds for each exercise.
# These values are intentionally conservative so the calorie estimate stays
# proportional to the detected movement rather than the raw frame rate.
AVG_SECONDS_PER_REP = {
    "bicep_curl": 2.5,
    "leg raises": 2.0,
    "pull Up": 2.8,
    "push-up": 2.3,
    "squat": 2.4,
    "tricep dips": 2.6,
}

MANUAL_EXERCISE_METRICS = {
    "bicep curl": (EXERCISE_METS["bicep_curl"], AVG_SECONDS_PER_REP["bicep_curl"]),
    "leg raises": (EXERCISE_METS["leg raises"], AVG_SECONDS_PER_REP["leg raises"]),
    "plank": (EXERCISE_METS["plank"], 2.0),
    "pull-up": (EXERCISE_METS["pull Up"], AVG_SECONDS_PER_REP["pull Up"]),
    "push-up": (EXERCISE_METS["push-up"], AVG_SECONDS_PER_REP["push-up"]),
    "squat": (EXERCISE_METS["squat"], AVG_SECONDS_PER_REP["squat"]),
    "tricep dips": (EXERCISE_METS["tricep dips"], AVG_SECONDS_PER_REP["tricep dips"]),
    "lunge": (5.0, 2.5),
}


def estimate_rep_calories(exercise, reps, weight_kg):
    """Estimate calories from exercise MET, body weight, and active rep time."""
    exercise_metrics = MANUAL_EXERCISE_METRICS.get(exercise.strip().casefold())
    if exercise_metrics is None:
        raise ValueError(f"Unsupported exercise for calorie estimation: {exercise}")
    if reps < 0 or weight_kg <= 0:
        raise ValueError("Reps must be non-negative and body weight must be positive.")

    met, seconds_per_rep = exercise_metrics
    return (met * float(weight_kg) * seconds_per_rep * int(reps)) / 3600.0


def estimate_timer_calories(exercise, seconds, weight_kg):
    """Estimate calories from an exercise hold duration, body weight, and MET."""
    exercise_metrics = MANUAL_EXERCISE_METRICS.get(exercise.strip().casefold())
    if exercise_metrics is None:
        raise ValueError(f"Unsupported exercise for calorie estimation: {exercise}")
    if seconds < 0 or weight_kg <= 0:
        raise ValueError("Seconds must be non-negative and body weight must be positive.")

    met, _ = exercise_metrics
    return (met * float(weight_kg) * float(seconds)) / 3600.0


class CalorieEstimator:
    """Estimate burned calories using MET values and incremental rep/timer counts."""

    def __init__(self, weight_kg=70.0, default_met=4.0):
        self.weight_kg = float(weight_kg)
        self.default_met = float(default_met)
        self.total_calories = 0.0
        self.current_exercise = None
        self.last_value = 0.0
        self.last_timestamp = time.monotonic()

    def _get_met(self, exercise):
        return EXERCISE_METS.get(exercise, self.default_met)

    def _reset_tracking(self, exercise, value):
        self.current_exercise = exercise
        self.last_value = float(value)
        self.last_timestamp = time.monotonic()

    def update(self, exercise, rep_result):
        """
        exercise: detector label, e.g. 'squat' or 'plank'
        rep_result: dict returned from RepCounter.update(), e.g.
            {'mode': 'reps', 'value': 5} or {'mode': 'timer', 'value': 12.3}

        Returns the cumulative estimated calories in kcal.
        """
        if exercise in (None, "unknown", "collecting..."): 
            return self.total_calories

        mode = rep_result.get("mode", "unknown")
        value = float(rep_result.get("value", 0.0))

        if exercise != self.current_exercise:
            self._reset_tracking(exercise, value)
            return self.total_calories

        delta = max(0.0, value - self.last_value)
        if delta <= 0:
            self.last_value = value
            return self.total_calories

        met = self._get_met(exercise)

        if mode == "reps":
            rep_seconds = AVG_SECONDS_PER_REP.get(exercise, 2.5) * delta
            self.total_calories += (met * self.weight_kg * rep_seconds) / 3600.0
        elif mode == "timer":
            self.total_calories += (met * self.weight_kg * delta) / 3600.0

        self.last_value = value
        return self.total_calories


if __name__ == "__main__":
    estimator = CalorieEstimator(weight_kg=68.0)
    print(estimator.update("squat", {"mode": "reps", "value": 10}))
    print(estimator.update("squat", {"mode": "reps", "value": 15}))
    print(estimator.update("plank", {"mode": "timer", "value": 20}))
