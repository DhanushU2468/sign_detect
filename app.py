import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)
import numpy as np
import pyttsx3
import threading
import av
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from gesture_utils import classify_gesture


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="YOLO Hand Pose + Voice", layout="wide")
st.title("🖐 Live Hand Gesture Detection + Voice Output (YOLO Pose, no OpenCV)")
st.write(
    "This version uses **YOLO pose + PIL** (no OpenCV / no MediaPipe), "
    "so it is compatible with Streamlit Cloud (Python 3.13)."
)


# ---------------- YOLO MODEL ----------------
@st.cache_resource
def load_yolo():
    # yolov8n-pose.pt will auto-download if not present
    return YOLO("yolov8n-pose.pt")

model = load_yolo()


# ---------------- VIDEO PROCESSOR ----------------
class HandPoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_spoken = None

    # ----- NON-BLOCKING VOICE -----
    def speak_async(self, text):
        def speech_job():
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()

        threading.Thread(target=speech_job, daemon=True).start()

    # ----- FRAME PROCESSING -----
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Get frame as RGB numpy array
        img_rgb = frame.to_ndarray(format="rgb24")

        # Run YOLO pose model
        results = model(img_rgb, verbose=False)

        # Convert numpy to PIL for drawing
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        gesture = None

        for result in results:
            if result.keypoints is None:
                continue

            # keypoints.xy: (num_objects, num_kpts, 2)
            kpts_xy = result.keypoints.xy
            if kpts_xy is None or len(kpts_xy) == 0:
                continue

            # For simplicity, take the first detected object
            kpts = kpts_xy[0].cpu().numpy()  # shape (K, 2)
            num_kpts = kpts.shape[0]

            # Build (K, 3) array: x, y, conf (dummy 1.0)
            kpts_full = np.zeros((num_kpts, 3), dtype=float)
            for i, (x, y) in enumerate(kpts):
                kpts_full[i] = [x, y, 1.0]
                # Minimal drawing: small green circle
                r = 3
                draw.ellipse(
                    (x - r, y - r, x + r, y + r),
                    fill=(0, 255, 0),
                    outline=None,
                )

            # Classify gesture from keypoints
            if num_kpts >= 21:
                # If your model has 21 keypoints (hand pose)
                gesture = classify_gesture(kpts_full)
            else:
                # If using body-pose model (17 kpts), you may want a different logic
                gesture = None

        # Voice and text overlay
        if gesture and gesture != self.last_spoken:
            self.speak_async(gesture)
            self.last_spoken = gesture

        if gesture:
            # Draw gesture label at top-left
            draw.text((20, 20), gesture, fill=(255, 0, 0))

        # Back to numpy
        out_frame = np.array(pil_img)
        return av.VideoFrame.from_ndarray(out_frame, format="rgb24")


# ---------------- WEBRTC CONFIG ----------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="yolo-hand-voice-no-opencv",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=HandPoseProcessor,
    async_processing=True,
)
