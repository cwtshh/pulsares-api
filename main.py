from fastapi import FastAPI, File, UploadFile
from moviepy.editor import VideoFileClip
from fastapi.responses import JSONResponse
import shutil
import os
import whisper

app = FastAPI()

os.makedirs("uploads", exist_ok=True)

@app.get('/')
def test_route():
    return {'message': 'Hello World'}


@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    # Salvar o arquivo no disco
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    return JSONResponse(content={"filename": file.filename})


def video_to_audio(file_location):
    video = VideoFileClip(file_location)
    audio = video.audio
    audio_file_location = file_location.rsplit('.', 1)[0] + '.wav'
    audio.write_audiofile(audio_file_location)
    return audio_file_location


