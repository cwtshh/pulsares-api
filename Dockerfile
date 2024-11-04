FROM python:3.12

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements_2.txt

RUN apt-get update && apt-get install -y ffmpeg

COPY . .

EXPOSE 8000

CMD ["fastapi", "dev", "cuda.py"]

