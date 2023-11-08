from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import time

app = FastAPI()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Enable CORS to allow requests from different origins (e.g., your friend's JavaScript app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow only trusted origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define an endpoint for real-time face detection
@app.get("/")
async def realtime_face_detection():
    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)  # Adjust the sleep time for the desired frame rate

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
