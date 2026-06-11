FROM python:3.12-slim
RUN pip install --no-cache-dir aiohttp==3.10.11
WORKDIR /app
COPY shim.py /app/shim.py
EXPOSE 8191
CMD ["python3", "/app/shim.py"]
