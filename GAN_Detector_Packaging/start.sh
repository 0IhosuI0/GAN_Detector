#!/bin/bash

if command -v nvidia-smi &> /dev/null
then
    echo "[System] NVIDIA GPU Detected! Starting in GPU mode..."
    docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
else
    echo "[System] No GPU detected. Starting in CPU mode...."
    docker-compose -f docker-compose.yml up --build -d
fi

echo "[System] Waiting 10 seconds for server initialization..."
sleep 10

echo "[System] Launching web browser..."

if [[ "$OSTYPE" == "darwin"* ]]; then

    open http://localhost:8000
elif command -v xdg-open &> /dev/null; then

    xdg-open http://localhost:8000
else
    echo "[System] Headless environment detected. Cannot open browser automatically."
    echo "[System] Please visit http://localhost:8000 manually."
fi