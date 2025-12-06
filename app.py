import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import cv2
import numpy as np
import pyttsx3
import threading
import av
from ultralytics import YOLO

from gesture_utils import classify_gesture


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="YOLO Hand Pose + Voice", layout="wide")
st.title("🖐 Live Hand Gesture Detection + Voice Output (YOLO Pose)")
st.write("Works on Streamlit Cloud (NO Mediapipe).")


# ---------------- YOLO MODEL ----------------
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n-pose.pt")   # Auto-downloads

model = load_yolo()


# ---------------- VIDEO PROCESSOR ----------------
class HandPoseProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_spoken = None

    def speak_async(self, text):

        def speech():
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()

        threading.Thread(target=speech, daemon=True).start()

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Run YOLO pose
        results = model(img, verbose=False)

        gesture = None

        for result in results:
            if result.keypoints is not None:

                kpts = result.keypoints.xy.cpu().numpy()[0]  # shape (21,2)

                # Convert single keypoints array to (21,3)
                kpts_full = np.zeros((21, 3))
                for i, p in enumerate(kpts):
                    kpts_full[i] = [p[0], p[1], 1]

                # Draw keypoints
                for (x, y, conf) in kpts_full:
                    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

                # Gesture
                gesture = classify_gesture(kpts_full)

        # Speak voice
        if gesture and gesture != self.last_spoken:
            self.speak_async(gesture)
            self.last_spoken = gesture

        # Display gesture
        if gesture:
            cv2.putText(img, gesture, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (255, 0, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- WEBRTC ----------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="yolo-hand-voice",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=HandPoseProcessor,
    async_processing=True,
)
