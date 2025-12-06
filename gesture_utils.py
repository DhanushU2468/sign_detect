import math


def classify_gesture(hand_landmarks):
    lm = hand_landmarks.landmark

    # Helper to check finger up/down
    def finger_up(tip, pip):
        return lm[tip].y < lm[pip].y

    # FINGER STATES
    thumb_up = lm[4].y < lm[3].y
    thumb_down = lm[4].y > lm[3].y
    index_up = finger_up(8, 6)
    middle_up = finger_up(12, 10)
    ring_up = finger_up(16, 14)
    pinky_up = finger_up(20, 18)

    fingers = [thumb_up, index_up, middle_up, ring_up, pinky_up]

    # ---- 1. STOP ✋ (all fingers up)
    if index_up and middle_up and ring_up and pinky_up:
        return "STOP"

    # ---- 2. THUMBS UP 👍
    if thumb_up and not index_up and not middle_up:
        return "THUMBS UP"

    # ---- 3. THUMBS DOWN 👎
    if thumb_down and not index_up and not middle_up:
        return "THUMBS DOWN"

    # ---- 4. OK 👌
    dist_ok = math.dist((lm[4].x, lm[4].y), (lm[8].x, lm[8].y))
    if dist_ok < 0.05:
        return "OK"

    # ---- 5. PEACE ✌️ (index + middle up)
    if index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"

    # ---- 6. ROCK 🤘 (index + pinky up)
    if index_up and pinky_up and not middle_up:
        return "ROCK"

    # ---- 7. FIST ✊ (all fingers down)
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "FIST"

    # ---- 8. OPEN PALM 🖐️ (4 fingers up, thumb neutral)
    if index_up and middle_up and ring_up and pinky_up and thumb_up:
        return "OPEN PALM"

    return None
