import os
import threading

import av
import numpy as np
import onnxruntime as ort
import pyttsx3
import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    VideoProcessorBase,
    webrtc_streamer,
)

from gesture_utils import classify_gesture


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Hand Gesture + Voice (ONNX)", layout="wide")
st.title("🖐 Live Hand Gesture Detection + Voice (ONNX, Cloud-Safe)")
st.write(
    "Runs a MediaPipe-style hand landmark ONNX model with **onnxruntime + PIL** "
    "(no OpenCV, no MediaPipe, no YOLO), so it works on Streamlit Cloud."
)


# ---------------- MODEL LOADING ----------------
MODEL_PATH = "hand_landmark.onnx"
ONNX_URL = (
    "https://github.com/PINTO0309/PINTO_model_zoo/"
    "raw/main/033_mediapipe_hand_landmark/hand_landmark_3d.onnx"
)
INPUT_SIZE = 256  # model expects 256x256 RGB


@st.cache_resource
def load_onnx_session():
    # Try to download model if not present
    if not os.path.exists(MODEL_PATH):
        try:
            st.info("Downloading ONNX hand landmark model...")
            r = requests.get(ONNX_URL, timeout=15)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
            st.success("Model downloaded successfully.")
        except Exception as e:
            st.error(
                "Could not download hand_landmark.onnx automatically.\n"
                "Please download it manually from PINTO_model_zoo "
                "-> 033_mediapipe_hand_landmark, rename to 'hand_landmark.onnx' "
                "and place it in the app folder.\n\n"
                f"Error: {e}"
            )
            raise

    sess = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    return sess, input_name


onnx_sess, onnx_input_name = load_onnx_session()


def run_hand_landmark(rgb_frame: np.ndarray) -> np.ndarray | None:
    """
    rgb_frame: (H, W, 3) uint8
    Returns: (21, 3) landmarks in normalized coords (0-1), or None.
    """
    # Resize to model input (256x256)
    pil_img = Image.fromarray(rgb_frame)
    pil_resized = pil_img.resize((INPUT_SIZE, INPUT_SIZE))
    img_np = np.asarray(pil_resized).astype(np.float32) / 255.0  # [0,1]
    img_np = img_np[np.newaxis, ...]  # (1, 256, 256, 3)

    outputs = onnx_sess.run(None, {onnx_input_name: img_np})
    if len(outputs) == 0:
        return None

    # Assume first output is (1, 63) -> 21*3
    lm = outputs[0].reshape(-1, 3)  # (21,3)
    if lm.shape[0] < 21:
        return None

    return lm  # normalized coords in model space


# ---------------- VIDEO PROCESSOR ----------------
class HandGestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_spoken = None

    def speak_async(self, text: str):
        """Speak text in a background thread. On cloud you won't hear it,
        but locally this will work."""
        def worker():
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                # On Streamlit Cloud audio may not be available – fail silently
                pass

        threading.Thread(target=worker, daemon=True).start()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Get frame as RGB
        img_rgb = frame.to_ndarray(format="rgb24")

        # Run ONNX landmark model
        landmarks = run_hand_landmark(img_rgb)
        gesture = None

        # Work on a 256x256 image for drawing (same as model input)
        pil_out = Image.fromarray(img_rgb).resize((INPUT_SIZE, INPUT_SIZE))
        draw = ImageDraw.Draw(pil_out)

        if landmarks is not None:
            # Draw minimal green dots for 21 keypoints
            for x_norm, y_norm, _ in landmarks:
                x = float(x_norm) * INPUT_SIZE
                y = float(y_norm) * INPUT_SIZE
                r = 3
                draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0))

            # Classify gesture in normalized model space
            gesture = classify_gesture(landmarks)

        # Voice: speak only when gesture changes
        if gesture and gesture != self.last_spoken:
            self.speak_async(gesture)
            self.last_spoken = gesture

        # Display label
        if gesture:
            draw.text((10, 10), gesture, fill=(255, 0, 0))

        # Back to numpy for WebRTC
        out_frame = np.array(pil_out)
        return av.VideoFrame.from_ndarray(out_frame, format="rgb24")


# ---------------- WEBRTC CONFIG ----------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="onnx-hand-gesture-voice",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=HandGestureProcessor,
    async_processing=True,
)
