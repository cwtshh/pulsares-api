import whisperx 
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import time
import subprocess
import torch, gc

os.makedirs("uploads", exist_ok=True)


device = "cuda"
batch_size = 8
compute_type = "float16"
model = whisperx.load_model("small", device, compute_type=compute_type)

app = FastAPI()

origins = [
    "http://localhost:5173",  # Adicione seu domínio aqui, se necessário
    "http://localhost:8000",
    "https://stellar.cwtsh.site",
    "https://stellar.aidadpdf.cloud"  # Ou o que for relevante
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permita apenas os domínios especificados
    allow_credentials=True,
    allow_methods=["*"],  # Permita todos os métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permita todos os cabeçalhos
)

@app.get('/')
async def test():
    return {"message": "Hello World"}

@app.post('/upload-video')
async def upload_video(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"

    with open(file_location, "wb") as local_file:
        local_file.write(await file.read())

    start_time = time.time()
    audio_path = convert_video_to_wav(file_location)
    conversion_time = time.time() - start_time

    if audio_path is None or not os.path.exists(audio_path):
        raise ValueError("Erro na conversão do vídeo para áudio. Caminho do áudio é inválido.")
    
    start_time = time.time()
    result = transcribe_audio_with_stamps(audio_path)
    transcription_time = time.time() - start_time

    os.remove(file_location)
    os.remove(audio_path)

    return JSONResponse(
        content={
            "filename": file.filename,
            "conversion_time": conversion_time,
            "transcription_time": transcription_time,
            "result": result
        }
    )


def convert_video_to_wav(video_path, output_path=None):
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + ".wav"
    else:
        output_path = os.path.join(output_path, os.path.basename(os.path.splitext(video_path)[0] + ".wav"))

    try:
        command = ["ffmpeg", "-i", video_path, output_path]
        subprocess.run(command, check=True)
        print(f"Converted {video_path} to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")

def transcribe_audio_with_stamps(audio_path):
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)
    diarize_segments = diarize_model(audio, min_speakers=4)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    gc.collect()
    torch.cuda.empty_cache()

    return result




