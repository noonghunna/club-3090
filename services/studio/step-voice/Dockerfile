# Step-Audio-EditX premium voice service — ISOLATED from ComfyUI.
# Step-Audio-EditX hard-pins transformers==4.53.3 (4.54+ = silent audio, per the model authors),
# which conflicts with ComfyUI's transformers 5.x (HiDream needs 5.x). So it runs in its OWN
# container with the exact transformers it was validated against — zero conflict, guaranteed-correct.
# Mirrors the Kokoro studio-tts service shape (HTTP server, writes to the shared gallery output dir).
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Vendor the reviewed Step-Audio-EditX inference (bundled step_audio_impl: tokenizer + TTS +
# CosyVoice2 vocoder + funasr_detach). Pinned to the commit we source-reviewed.
ARG NODE_REF=d8eddc6618fdbe14a10cc99d13360a64a2b03657
RUN git clone https://github.com/Saganaki22/ComfyUI-Step_Audio_EditX_TTS.git /opt/step \
    && git -C /opt/step checkout ${NODE_REF}

# The repo's requirements.txt pins transformers==4.53.3 + the audio/whisper/onnx deps. torch +
# torchaudio come from the base image (satisfy the >= floors). gradio is unused by the server but
# harmless. + aiohttp for the HTTP layer.
RUN pip install --no-cache-dir -r /opt/step/requirements.txt
# Deps the repo's requirements.txt omits but the CosyVoice2 vocoder needs (found at model-load):
# einops (decoder_dit), ffmpeg-python (audio I/O), rotary_embedding_torch (mossformer path). + aiohttp.
RUN pip install --no-cache-dir aiohttp einops ffmpeg-python rotary_embedding_torch

COPY server.py /opt/server.py

ENV STEP_IMPL=/opt/step/step_audio_impl \
    STEP_MODELS_DIR=/models \
    OUTPUT_DIR=/output \
    VOICE_SAMPLES_DIR=/opt/step/voice_samples \
    STEP_VOICE_PORT=8193 \
    STEP_DEFAULT_VOICE=Narrator.wav \
    STEP_TORCH_DTYPE=bfloat16 \
    WHISPER_MODEL=base
WORKDIR /opt
EXPOSE 8193
CMD ["python3", "/opt/server.py"]
