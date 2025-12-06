# mediapipe_model.py
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection


class HandDetector:
    def __init__(self):
        print("🔄 Loading MediaPipe Hand Detection model from Hugging Face...")
        self.processor = AutoImageProcessor.from_pretrained(
            "qualcomm/MediaPipe-Hand-Detection"
        )
        self.model = AutoModelForObjectDetection.from_pretrained(
            "qualcomm/MediaPipe-Hand-Detection"
        )
        print("✅ Model Loaded Successfully")

    def predict(self, image):
        """
        image: PIL.Image in RGB mode
        returns: dict with 'scores', 'labels', 'boxes'
        """
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=0.3,
            target_sizes=target_sizes
        )[0]

        return results
