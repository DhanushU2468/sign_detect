# inference.py
from PIL import Image

from mediapipe_model import HandDetector

# create a single global detector instance
detector = HandDetector()

def run_detection(image: Image):
    """
    image: PIL RGB
    returns: list of predictions with score, label, box
    """
    results = detector.predict(image)
    predictions = []

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        if float(score) > 0.3:
            predictions.append({
                "score": float(score),
                "label": int(label),
                "box": [float(x) for x in box]
            })

    return predictions
