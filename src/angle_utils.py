import numpy as np


def calculate_angle(a, b, c):
    """
    Calculates the angle at point b, given three (x, y) points.
    a, b, c: tuples/lists like (x, y)
    Returns angle in degrees (0-180). 
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def get_angle_vector(landmarks):
    """
    landmarks: list of 33 tuples (x_px, y_px, z, visibility),
    as returned by PoseDetector.get_landmark_list() — works for both
    the old mp.solutions.pose API and the mediapipe.tasks (0.10.11) API,
    since both produce the same (x, y, z, visibility) tuple format.

    Returns an 8-element feature vector:
    [l_elbow, r_elbow, l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee]
    """
    def pt(i):
        return (landmarks[i][0], landmarks[i][1])  # (x, y) only

    l_elbow    = calculate_angle(pt(11), pt(13), pt(15))   # shoulder-elbow-wrist
    r_elbow    = calculate_angle(pt(12), pt(14), pt(16))
    l_shoulder = calculate_angle(pt(13), pt(11), pt(23))   # elbow-shoulder-hip
    r_shoulder = calculate_angle(pt(14), pt(12), pt(24))
    l_hip      = calculate_angle(pt(11), pt(23), pt(25))   # shoulder-hip-knee
    r_hip      = calculate_angle(pt(12), pt(24), pt(26))
    l_knee     = calculate_angle(pt(23), pt(25), pt(27))   # hip-knee-ankle
    r_knee     = calculate_angle(pt(24), pt(26), pt(28))

    return np.array([l_elbow, r_elbow, l_shoulder, r_shoulder,
                      l_hip, r_hip, l_knee, r_knee])
