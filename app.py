import threading

import av
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import streamlit as st
from streamlit_webrtc import (RTCConfiguration, VideoProcessorBase, WebRtcMode,
                              webrtc_streamer)

from gesture_utils import classify_gesture

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="Hand Gesture + Voice", layout="wide")

st.title("🖐 Live Hand Gesture Detection + Voice Output")
st.write("Built with **MediaPipe + Streamlit WebRTC + Non-blocking pyttsx3**.")


# ---------------------- MEDIAPIPE SETUP ----------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils



# ---------------------- VIDEO PROCESSOR ----------------------
class HandTrackingProcessor(VideoProcessorBase):

    def __init__(self):
        # Initialize Mediapipe
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        # Track last spoken gesture
        self.last_spoken = None
        self.speaking_thread = None


    def speak_non_blocking(self, text):
        """Speak gesture WITHOUT freezing webcam using separate thread."""

        def speech_job():
            engine = pyttsx3.init()          # Fresh engine each time (important)
            engine.setProperty("rate", 150)  # Speed
            engine.say(text)
            engine.runAndWait()
            engine.stop()

        # Start async speech
        threading.Thread(target=speech_job, daemon=True).start()


    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)
        gesture = None

        # Draw and classify gesture
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                gesture = classify_gesture(hand_landmarks)

        # Speak only when gesture CHANGES
        if gesture and gesture != self.last_spoken:
            self.speak_non_blocking(gesture)
            self.last_spoken = gesture

        # Show gesture on video output
        if gesture:
            cv2.putText(img, gesture, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")



# ---------------------- WEBRTC CONFIG ----------------------
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)



# ---------------------- START WEBCAM ----------------------
webrtc_streamer(
    key="gesture-voice-final",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=HandTrackingProcessor,
    async_processing=True,
)
