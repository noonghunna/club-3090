FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN python -m pip install \
        "litellm[proxy]>=1.80" \
        "protobuf>=5" \
        "sentencepiece>=0.2" \
        "tiktoken>=0.9" \
        "transformers>=4.51"

WORKDIR /app
COPY router/ /app/router/

EXPOSE 8000

ENTRYPOINT ["litellm"]
CMD ["--config", "/app/router/config.quad-pairs.yaml", "--host", "0.0.0.0", "--port", "8000"]
