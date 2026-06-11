FROM python:3.12-slim
# espeak-ng = Kokoro G2P fallback; ffmpeg = mixdown; libsndfile1 = soundfile WAV I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
# kokoro-onnx pulls onnxruntime (CPU) + misaki (G2P). Pin onnxruntime to the CPU wheel.
RUN pip install --no-cache-dir kokoro-onnx soundfile aiohttp
WORKDIR /app
COPY tts.py /app/tts.py
EXPOSE 8192
CMD ["python3", "/app/tts.py"]
