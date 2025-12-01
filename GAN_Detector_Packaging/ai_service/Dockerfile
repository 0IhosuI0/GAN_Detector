FROM python:3.9-slim

WORKDIR /app

#RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models ./models/
COPY haarcascade_frontalface_default.xml .
COPY ai_server.py .
RUN wget -O models/ResNet_EPOCH100.h5 "https://github.com/ghtn10000/GAN_Detector/releases/download/v1.0.1/ResNet_EPOCH100.h5"
RUN wget -O models/teacher.keras "https://github.com/ghtn10000/GAN_Detector/releases/download/v1.0.1/teacher.keras"
RUN wget -O models/student.keras "https://github.com/ghtn10000/GAN_Detector/releases/download/v1.0.1/student.keras"
RUN wget -O models/classifier.keras "https://github.com/ghtn10000/GAN_Detector/releases/download/v1.0.1/classifier.keras"

CMD ["uvicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "35480"]