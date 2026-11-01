import datetime
import hashlib
import os
from pathlib import Path
import uuid

try:
    from google.api_core.exceptions import AlreadyExists
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:  # pragma: no cover - optional Firebase dependency in demo mode
    AlreadyExists = None
    firebase_admin = None
    credentials = None
    firestore = None


class DuplicateUserError(Exception):
    """Raised when an email address or username is already registered."""


def init_db():
    """Initialize and return the Firebase Cloud Firestore client.

    When Firebase Admin SDK dependencies are unavailable or credentials are not
    configured, return None so the app can still run in local demo mode.
    """
    if firebase_admin is None or firestore is None or credentials is None:
        print("Firebase Admin SDK unavailable; running in demo mode without database logging.")
        return None

    credential_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credential_path:
        print("GOOGLE_APPLICATION_CREDENTIALS not set; running in demo mode without database logging.")
        return None

    key_file = Path(credential_path).expanduser()
    if not key_file.is_file():
        print(f"Firebase service-account file was not found: {key_file}; running in demo mode without database logging.")
        return None

    try:
        firebase_app = firebase_admin.get_app()
    except ValueError:
        firebase_app = firebase_admin.initialize_app(
            credentials.Certificate(str(key_file))
        )
    return firestore.client(app=firebase_app)


def _normalized_key(value):
    return value.strip().casefold()


def user_index_id(value):
    """Return a stable Firestore-safe ID for an email or username index."""
    return hashlib.sha256(_normalized_key(value).encode("utf-8")).hexdigest()


def get_user_by_id(db, user_id):
    snapshot = db.collection("users").document(str(user_id)).get()
    if not snapshot.exists:
        return None
    return {"id": snapshot.id, **snapshot.to_dict()}


def get_user_by_email(db, email):
    index = db.collection("user_emails").document(user_index_id(email)).get()
    if not index.exists:
        return None
    return get_user_by_id(db, index.get("user_id"))


def create_user(db, username, email, password_hash):
    if db is None:
        raise RuntimeError("Firebase database is unavailable in demo mode.")

    user_id = uuid.uuid4().hex
    user_ref = db.collection("users").document(user_id)
    email_ref = db.collection("user_emails").document(user_index_id(email))
    username_ref = db.collection("usernames").document(user_index_id(username))
    user_data = {
        "username": username,
        "email": email,
        "password_hash": password_hash,
        "gender": None,
        "weight_kg": None,
        "default_rest_seconds": 90,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    batch = db.batch()
    batch.create(user_ref, user_data)
    batch.create(email_ref, {"user_id": user_id})
    batch.create(username_ref, {"user_id": user_id})
    try:
        batch.commit()
    except AlreadyExists as error:
        raise DuplicateUserError from error

    return {"id": user_id, **user_data}


def update_user_profile(db, user_id, gender, weight_kg, default_rest_seconds=90):
    db.collection("users").document(str(user_id)).update({
        "gender": gender,
        "weight_kg": weight_kg,
        "default_rest_seconds": int(default_rest_seconds),
    })


def log_weight_entry(db, user_id, weight_kg, goal):
    """Store one weight check-in and its selected fitness goal."""
    if db is None:
        raise RuntimeError("Firestore is unavailable; weight entries cannot be saved.")

    entry = {
        "user_id": str(user_id),
        "weight_kg": float(weight_kg),
        "goal": goal,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    reference = db.collection("weight_entries").document()
    batch = db.batch()
    batch.set(reference, entry)
    batch.update(
        db.collection("users").document(str(user_id)),
        {"weight_kg": entry["weight_kg"]},
    )
    batch.commit()
    return {"id": reference.id, **entry}


def get_weight_entries(db, user_id):
    """Return one user's weight check-ins in chronological order."""
    if db is None:
        return []

    query = db.collection("weight_entries").where("user_id", "==", str(user_id))
    entries = [
        {"id": snapshot.id, **snapshot.to_dict()}
        for snapshot in query.stream()
    ]
    return sorted(entries, key=lambda entry: entry.get("timestamp", ""))


def get_weekly_plan(db, user_id):
    days = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    if db is None:
        return {day: [] for day in days}

    snapshot = db.collection("weekly_workout_plans").document(str(user_id)).get()
    if not snapshot.exists:
        return {day: [] for day in days}

    saved_plan = snapshot.to_dict().get("days", {})
    return {
        day: saved_plan.get(day, [])
        if isinstance(saved_plan.get(day, []), list)
        else []
        for day in days
    }


def save_weekly_plan(db, user_id, weekly_plan):
    if db is None:
        raise RuntimeError("Firestore is unavailable; workout plans cannot be saved.")

    db.collection("weekly_workout_plans").document(str(user_id)).set({
        "user_id": str(user_id),
        "days": weekly_plan,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })


def save_feedback(db, user_id, username, category, message):
    """Store one signed-in user's app feedback in Firestore."""
    if db is None:
        raise RuntimeError("Firestore is unavailable; feedback cannot be saved.")

    feedback = {
        "user_id": str(user_id),
        "username": username,
        "category": category,
        "message": message,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    db.collection("user_feedback").document().set(feedback)


def log_session(
    db,
    exercise,
    mode,
    reps,
    hold_seconds,
    duration_sec,
    calories,
    user_id=None,
    sets=None,
    workout_day=None,
    volume_kg=None,
    notes=None,
    rpe=None,
):
    """Insert one completed exercise session into Firestore."""
    if db is None:
        return

    session = {
        "user_id": str(user_id) if user_id is not None else None,
        "exercise": exercise,
        "mode": mode,
        "reps": reps,
        "hold_seconds": hold_seconds,
        "duration_sec": duration_sec,
        "calories": float(calories or 0),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if sets is not None:
        session["sets"] = sets
        session["workout_day"] = workout_day
        session["volume_kg"] = float(volume_kg or 0)
        session["notes"] = notes or ""
        session["rpe"] = rpe
    db.collection("workout_sessions").document().set(session)


def _session_tuple(snapshot):
    session = snapshot.to_dict()
    return (
        snapshot.id,
        session.get("user_id"),
        session.get("exercise"),
        session.get("mode"),
        session.get("reps"),
        session.get("hold_seconds"),
        session.get("duration_sec"),
        float(session.get("calories") or 0),
        session.get("timestamp"),
    )


def get_all_sessions(db, user_id=None):
    """Return sessions in the legacy tuple format, newest first."""
    query = db.collection("workout_sessions")
    if user_id is not None:
        query = query.where("user_id", "==", str(user_id))
    sessions = [_session_tuple(snapshot) for snapshot in query.stream()]
    return sorted(sessions, key=lambda row: row[8] or "", reverse=True)


def get_manual_session_details(db, user_id):
    """Return saved set details for manual sessions owned by one user."""
    if db is None:
        return {}

    query = db.collection("workout_sessions").where("user_id", "==", str(user_id))
    details = {}
    for snapshot in query.stream():
        session = snapshot.to_dict()
        if session.get("mode") == "manual":
            details[snapshot.id] = {
                "sets": session.get("sets", []),
                "volume_kg": float(session.get("volume_kg") or 0),
                "workout_day": session.get("workout_day"),
                "notes": session.get("notes", ""),
                "rpe": session.get("rpe"),
            }
    return details


def clear_user_sessions(db, user_id):
    """Delete and return the number of workout sessions belonging to one user."""
    collection = db.collection("workout_sessions")
    query = collection.where("user_id", "==", str(user_id))
    snapshots = list(query.stream())

    for offset in range(0, len(snapshots), 500):
        batch = db.batch()
        for snapshot in snapshots[offset:offset + 500]:
            batch.delete(collection.document(snapshot.id))
        batch.commit()

    return len(snapshots)


def delete_user_session(db, session_id, user_id):
    """Delete one workout session only when it belongs to the given user."""
    session_ref = db.collection("workout_sessions").document(str(session_id))
    snapshot = session_ref.get()
    if not snapshot.exists or str(snapshot.get("user_id")) != str(user_id):
        return False

    session_ref.delete()
    return True


def get_sessions_by_exercise(db, exercise, user_id=None):
    """Return sessions for one exercise, optionally limited to one user."""
    query = db.collection("workout_sessions").where("exercise", "==", exercise)
    if user_id is not None:
        query = query.where("user_id", "==", str(user_id))
    sessions = [_session_tuple(snapshot) for snapshot in query.stream()]
    return sorted(sessions, key=lambda row: row[8] or "", reverse=True)


def get_summary_stats(db, user_id=None):
    """Return per-exercise totals in the legacy tuple format."""
    totals = {}
    for session in get_all_sessions(db, user_id):
        exercise = session[2]
        if exercise not in totals:
            totals[exercise] = [0, 0, 0, 0]
        totals[exercise][0] += session[4] or 0
        totals[exercise][1] += session[5] or 0
        totals[exercise][2] += session[7] or 0
        totals[exercise][3] += 1
    return [
        (exercise, *totals[exercise])
        for exercise in sorted(totals, key=lambda value: str(value))
    ]
