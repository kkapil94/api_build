import cv2
from fastapi import FastAPI, File, UploadFile
import aiofiles
import os

app = FastAPI()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

async def process_video(file_path):
        result = []
        video_capture = cv2.VideoCapture(file_path)
        while True:
            ret, frame = video_capture.read()
            if not ret: 
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return faces.tolist()
            
@app.get("/")
def getApp():
    return {"file":"ok"}

@app.post("/video/detect-faces")
async def detect_faces(file: UploadFile = File(...)):
        try:
            async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp:
                contents = await file.read()
                await temp.write(contents)
            processed_results = await process_video(temp.name)
        except Exception:
            return {"message": "There was an error processing the file"}
        finally:
            os.remove(temp.name)

        return processed_results # Return the processed frames

