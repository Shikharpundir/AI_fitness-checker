def classify_exercise(angles):
    """
    angles: 8-element array from get_angle_vector()
    [l_elbow, r_elbow, l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee]

    Returns: exercise label as a string (must match EXERCISE_CONFIG keys in rep_counter.py)
    """
    l_elbow, r_elbow, l_sh, r_sh, l_hip, r_hip, l_knee, r_knee = angles 

    avg_elbow = (l_elbow + r_elbow) / 2
    avg_shoulder = (l_sh + r_sh) / 2
    avg_hip = (l_hip + r_hip) / 2
    avg_knee = (l_knee + r_knee) / 2

    # Plank: body horizontal, hip angle 160-190 degrees
    if 160 <= avg_hip <= 190 and avg_elbow < 120 and avg_knee > 160:
        return "plank"

    # Squat: knees and hips both bent significantly
    elif avg_knee < 140 and avg_hip < 140:
        return "squat"

    # Bicep curl: elbow bent (< 90), torso upright (hip > 150)
    elif avg_elbow < 90 and avg_hip > 150:
        return "bicep_curl"

    # Push-up: elbow bent (< 100), body horizontal, knees straight
    elif avg_elbow < 100 and 100 < avg_hip < 200 and avg_knee > 160:
        return "push-up"

    # Pull-up: elbow bent (< 90), shoulder raised, body vertical
    elif avg_elbow < 90 and avg_shoulder < 150 and avg_hip > 150:
        return "pull Up"

    # Tricep dips: elbow bent, shoulders raised, body vertical
    elif avg_elbow < 100 and avg_shoulder > 100 and avg_hip > 150:
        return "tricep dips"

    # Leg raises: hips bent (< 90), knees mostly straight
    elif avg_hip < 100 and avg_knee > 150:
        return "leg raises"

    else:
        return "unknown"
