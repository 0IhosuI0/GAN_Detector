FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./modles/
COPY haarcascade_frontalface_default.xml .
COPY ai_server.py .

CMD ["unicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "35840"]