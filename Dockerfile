# Build stage
FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    pkg-config \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
# Install dependencies using the built wheels
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
