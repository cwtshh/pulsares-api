FROM python:3.12

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN sudo apt update \
    sudo apt install ffmpeg

COPY . .

EXPOSE 8045

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8045", "--reload"]

