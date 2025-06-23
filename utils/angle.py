import numpy as np

def calculate_angle_cosine(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_average_joint_angle(keypoints, left_idxs, right_idxs):
    kp = keypoints.data[0]
    left_angle = calculate_angle_cosine(kp[left_idxs[0]], kp[left_idxs[1]], kp[left_idxs[2]])
    right_angle = calculate_angle_cosine(kp[right_idxs[0]], kp[right_idxs[1]], kp[right_idxs[2]])
    return (left_angle + right_angle) / 2

def get_form_feedback(angle, maintaining, relaxing, tolerance=10):
    if angle < (maintaining - tolerance):
        return "Too bent — try to extend more."
    elif angle > (relaxing + tolerance):
        return "Too extended — flex more for a full curl."
    elif abs(angle - maintaining) <= tolerance:
        return "Great form!"
    else:
        return "Almost there — maintain proper range."