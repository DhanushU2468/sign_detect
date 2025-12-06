import math
import numpy as np

def classify_gesture(kpts):
    """
    kpts = 21-keypoint array of shape (21,3) from YOLO pose
    Each keypoint: [x, y, confidence]
    """

    # Landmarks mapping (YOLO hand model)
    # Index reference same as MediaPipe:
    # 0: wrist, 4: thumb tip, 8: index tip, etc.

    def up(tip, pip):
        return kpts[tip][1] < kpts[pip][1]

    thumb_up = up(4, 3)
    thumb_down = kpts[4][1] > kpts[3][1]
    index_up = up(8, 6)
    middle_up = up(12, 10)
    ring_up = up(16, 14)
    pinky_up = up(20, 18)

    # STOP ✋
    if index_up and middle_up and ring_up and pinky_up:
        return "STOP"

    # THUMBS UP 👍
    if thumb_up and not index_up:
        return "THUMBS UP"

    # THUMBS DOWN 👎
    if thumb_down and not index_up:
        return "THUMBS DOWN"

    # OK 👌
    dist_ok = math.dist(kpts[4][:2], kpts[8][:2])
    if dist_ok < 25:
        return "OK"

    # PEACE ✌️
    if index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"

    # ROCK 🤘
    if index_up and pinky_up and not middle_up:
        return "ROCK"

    # FIST ✊
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "FIST"

    # OPEN PALM
    if index_up and middle_up and ring_up and pinky_up and thumb_up:
        return "OPEN PALM"

    return None
