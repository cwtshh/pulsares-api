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
import json
import logging
import shutil

from sympy.strategies.core import switch
from whisperx import align


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

relative_dir = os.path.join(os.getcwd(), "models")  # Define a relative directory path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_KEYS_FILE = "model_keys.json"  # Arquivo para persistência das chaves

# Função para salvar as chaves dos modelos em um arquivo JSON
def save_model_keys():
    with open(MODEL_KEYS_FILE, "w") as file:
        json.dump(list(models.keys()), file)

# Função para carregar as chaves dos modelos do arquivo JSON ao iniciar a API
def load_model_keys():
    global models
    if os.path.exists(MODEL_KEYS_FILE):
        with open(MODEL_KEYS_FILE, "r") as file:
            model_keys = json.load(file)
            for key in model_keys:
                models[key] = None  # Inicializa os modelos com None ou outro valor apropriado

load_model_keys()  # Carrega as chaves ao iniciar a API


@app.get("/list-models/")
def list_models():
    model_keys=[nome for nome in os.listdir(relative_dir) if os.path.isdir(os.path.join(relative_dir, nome))]
     # Retorna as chaves dos modelos carregados
    return {"models": model_keys}



# Endpoint para deletar um modelo específico
@app.delete("/delete-models/{model_key}")
def delete_model(model_key: str):
    if model_key in models: 
        model_dir = os.path.join(relative_dir, model_key) 
        del models[model_key]
        shutil.rmtree(model_dir)
        return {"message": f"Model {model_key} deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.get('/')
def test_route():
    return {'message': 'Hello World'}


@app.post("/define-model/")
def load_model(request: ModelRequest):
    global models
    key = f"{request.model_type}_{request.model_name}_{request.model_size}_{request.device}_{request.compute_type}"
    model_dir = os.path.join(relative_dir, key)  # Define o caminho da subpasta

    # Cria a pasta do modelo se ela ainda não existir
    os.makedirs(relative_dir, exist_ok=True)
    
    if key not in models:
        # Carrega o modelo baseado no tipo
        if request.model_type == "faster_whisper":
            model = WhisperModel(request.model_size, device=request.device, compute_type=request.compute_type, download_root=model_dir)
        elif request.model_type == "whisperx":
            model = whisperx.load_model(request.model_size, request.device, compute_type=request.compute_type, download_root=model_dir)
        else:
            raise ValueError("Invalid model type. Choose either 'faster_whisper' or 'whisperx'.")
       
        models[key] = model
        save_model_keys()  # Salva as chaves após o carregamento de um novo modelo
    else:
        model = models[key]
    
    return {"message": f"Model {key} loaded successfully in {relative_dir}."}


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
    model_key = f"{model_request.model_type}_{model_request.model_name}_{model_request.model_size}_{model_request.device}_{model_request.compute_type}"
    model = models[model_key]
    load_model(models)  # Load the model using the updated function

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    if model_request.model_type == "whisperx":
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=model_request.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, model_request.device, return_char_alignments=False)

        diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=model_request.device)
        diarize_segments = diarize_model(audio, min_speakers=4)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    return result

