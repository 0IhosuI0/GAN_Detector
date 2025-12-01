@echo off
setlocal

where nvidia-smi >nul 2>nul

if %ERRORLEVEL% equ 0 (
    echo [System] NVIDIA GPU Found! Running in GPU mode...
    echo ------------------------------------------------
    docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
) else (
    echo [System] GPU NOT found. Running in CPU mode...
    echo ------------------------------------------------
    docker-compose up --build -d
)

echo.
echo [System] Waiting for servers to initialize (10 seconds)...

timeout /t 10 /nobreak >nul

echo [System] Opening Web Browser...

start http://localhost:5050


echo.
echo [Done] Servers are running.
pause