# Tiny host-side orchestrator: stdlib Python + ffmpeg (no extra pip deps).
FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
COPY orchestrator.py /app/orchestrator.py
WORKDIR /app
CMD ["python3", "orchestrator.py"]
