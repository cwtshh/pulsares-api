FROM python:3.12

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y ffmpeg

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8045

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8045", "--reload"]

