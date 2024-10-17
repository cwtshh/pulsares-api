from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
import subprocess

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

    convert_video_to_wav(file_location)

    return JSONResponse(content={"filename": file.filename})



def convert_video_to_wav(video_path, output_path=None):

    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + '.wav'
    else:
        output_path = os.path.join(output_path, os.path.basename(os.path.splitext(video_path)[0] + '.wav'))

    try:
        command = ['ffmpeg', '-i', video_path, output_path]

        subprocess.run(command, check=True)
        print(f"Arquivo convertido com sucesso: {output_path}")

    except subprocess.CalledProcessError as e:
        print(f"Erro ao converter vídeo: {e}")


   