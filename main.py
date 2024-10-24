from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from faster_whisper import WhisperModel as whisper
import subprocess
import time

app = FastAPI()

os.makedirs("uploads", exist_ok=True)

origins = [
    "http://localhost:5173",  # Adicione seu domínio aqui, se necessário
    "http://localhost:8000",  # Ou o que for relevante
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permita apenas os domínios especificados
    allow_credentials=True,
    allow_methods=["*"],  # Permita todos os métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permita todos os cabeçalhos
)

@app.get('/')
def test_route():
    return {'message': 'Hello World'}

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    # Save the file to disk
    with open(file_location, "wb") as f:
        f.write(await file.read())

    start_time = time.time()  # Start the timer
    audio_path = convert_video_to_wav(file_location)
    conversion_time = time.time() - start_time  # Calculate conversion time

    start_time = time.time()  # Restart the timer
    result = transcribe_audio_with_stamps(audio_path)
    transcription_time = time.time() - start_time  # Calculate transcription time

    # Delete temporary files
    os.remove(file_location)
    os.remove(audio_path)

    # Convert the result to a list if it is a generator
    if isinstance(result, (list, tuple)):
        result_list = result
    else:
        result_list = list(result)

    return JSONResponse(content={
        "filename": file.filename,
        "conversion_time": conversion_time,
        "transcription_time": transcription_time,
        "result": result_list
    })

def convert_video_to_wav(video_path, output_path=None):
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + '.wav'
    else:
        output_path = os.path.join(output_path, os.path.basename(os.path.splitext(video_path)[0] + '.wav'))

    try:
        command = ['ffmpeg', '-i', video_path, output_path]
        subprocess.run(command, check=True)
        print(f"Arquivo convertido com sucesso: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Erro ao converter vídeo: {e}")

def transcribe_audio_with_stamps(audio_path):
    # Carregar o modelo uma vez
    model_size = "small"
    model = whisper(model_size, device="cpu", compute_type="int8")

    # Transcrever o áudio usando o modelo carregado
    result, info = model.transcribe(audio_path, beam_size=5)

    # Convert the generator to a list
    result_list = list(result)

    return result_list

