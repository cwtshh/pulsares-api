from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import subprocess
import whisperx
from pydantic import BaseModel
from faster_whisper import WhisperModel
import time
import torch,gc


app = FastAPI()
load_dotenv()
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

class ModelRequest(BaseModel):
    model_name: str
    model_type: str
    model_size: str
    device: str
    compute_type: str

models = {}
last_loaded_model_request = None

relative_dir = os.path.join(os.getcwd(), "models")  # Define a relative directory path

@app.get('/')
def test_route():
    return {'message': 'Hello World'}

@app.get("/models/")
def get_models():
    return list(models.keys())
@app.post("/define-model/")
def load_model(request: ModelRequest):
    global models
    key = f"{request.model_type}_{request.model_name}_{request.model_size}_{request.device}_{request.compute_type}"

    if key not in models:
        if request.model_type == "faster_whisper":
            model = WhisperModel(request.model_size, device=request.device, compute_type=request.compute_type, download_root=relative_dir)
        elif request.model_type == "whisperx":
            model = whisperx.load_model(request.model_size, request.device, compute_type=request.compute_type, download_root=relative_dir)
        else:
            raise ValueError("Invalid model type. Choose either 'faster_whisper' or 'whisperx'.")

        models[key] = model
    else:
        model = models[key]

    last_loaded_model_request = request
    return {"message": f"Modelo {request.model_name} do tipo {request.model_type} e tamanho {request.model_size} carregado com sucesso."}
@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    # Salvar o arquivo no disco
    with open(file_location, "wb") as f:
        f.write(await file.read())

    start_time = time.time()  # Iniciar o temporizador
    audio_path = convert_video_to_wav(file_location)
    conversion_time = time.time() - start_time  # Calcular tempo de conversão

    start_time = time.time()  # Reiniciar o temporizador
    result = transcribe_audio_with_stamps(audio_path)
    transcription_time = time.time() - start_time  # Calcular tempo de transcrição

    # deletar arquivos temporários
    os.remove(file_location)
    os.remove(audio_path)

    return JSONResponse(content={
        "filename": file.filename,
        "conversion_time": conversion_time,
        "transcription_time": transcription_time,
        "result": result
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
    global last_loaded_model_request

    if last_loaded_model_request is None:
        raise ValueError("Modelo ainda não carregado.")

    model_key = f"{last_loaded_model_request.model_type}_{last_loaded_model_request.model_name}_{last_loaded_model_request.model_size}_{last_loaded_model_request.device}_{last_loaded_model_request.compute_type}"
    model = models[model_key]

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    if last_loaded_model_request.model_type == "whisperx":
        model_a, metadata = whisperx.load_align_model(language_code=result["language"],
                                                      device=last_loaded_model_request.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, last_loaded_model_request.device,
                                return_char_alignments=False)

        diarize_model = whisperx.DiarizationPipeline(use_auth_token="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",
                                                     device=last_loaded_model_request.device)
        diarize_segments = diarize_model(audio, min_speakers=4)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    if last_loaded_model_request.device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    return result
