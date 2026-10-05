FROM python:3.11-slim

WORKDIR /app

# System deps needed by opencv-python-headless and Pillow's font rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

EXPOSE 8000

# Downloads the model from Blob Storage, then starts the API
CMD ["sh", "-c", "python src/download_model.py && uvicorn src.app:app --host 0.0.0.0 --port 8000"]