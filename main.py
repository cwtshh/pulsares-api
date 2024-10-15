from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os

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


