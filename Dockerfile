FROM python:3.9-alpine

RUN apk add --no-cache \
    ffmpeg

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
