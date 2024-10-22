from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import subprocess
import whisperx
import time
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

@app.get('/')
def test_route():
    return {'message': 'Hello World'}

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
    device = "cpu"

    # Carregar o modelo uma vez
    model = whisperx.load_model("small", device=device, compute_type="float32")
    audio = whisperx.load_audio(audio_path)

    # Assuming the correct function is `transcribe_audio`
    result = model.transcribe(audio)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    diarize_model = whisperx.DiarizationPipeline(device=device)
    diarize_model(audio, min_speakers=4)

    diarize_segments = diarize_model(audio)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result

