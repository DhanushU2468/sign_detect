import math
import numpy as np

def classify_gesture(kpts: np.ndarray):
    """
    kpts: numpy array of shape (21, 3)
          each row: [x, y, z_or_conf] in normalized coords (0–1)
    Returns a string label or None.
    """

    if kpts.shape[0] < 21:
        return None

    # y smaller = "higher" on screen
    def finger_up(tip_idx, pip_idx):
        return kpts[tip_idx][1] < kpts[pip_idx][1]

    thumb_up = kpts[4][1] < kpts[3][1]
    thumb_down = kpts[4][1] > kpts[3][1]
    index_up = finger_up(8, 6)
    middle_up = finger_up(12, 10)
    ring_up = finger_up(16, 14)
    pinky_up = finger_up(20, 18)

    # 1) STOP ✋ - all four fingers up
    if index_up and middle_up and ring_up and pinky_up:
        return "STOP"

    # 2) THUMBS UP 👍
    if thumb_up and not index_up and not middle_up:
        return "THUMBS UP"

    # 3) THUMBS DOWN 👎
    if thumb_down and not index_up and not middle_up:
        return "THUMBS DOWN"

    # 4) OK 👌 - thumb/index close
    dist_ok = math.dist(kpts[4][:2], kpts[8][:2])
    if dist_ok < 0.05:
        return "OK"

    # 5) PEACE ✌️ - index + middle up
    if index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"

    # 6) ROCK 🤘 - index + pinky up, middle down
    if index_up and pinky_up and not middle_up:
        return "ROCK"

    # 7) FIST ✊ - all fingers down
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "FIST"

    # 8) OPEN PALM 🖐️ - all fingers + thumb up
    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "OPEN PALM"

    return None
