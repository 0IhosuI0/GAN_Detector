from flask import Flask, jsonify, request, current_app
from flask_cors import CORS
import requests
import numpy as np
from PIL import Image
import io
import os
import logging
from logging.handlers import FileHandler

app = Flask(__name__)
CORS(app)

## 로그
file_handler = FileHandler('server.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

## 보안 더블 체크
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # 10MB 용량 제한

def allowed_file(filename):
    """파일 이름에 '.'이 있고, 허용된 확장자인지 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

## 내일 물어보고 정보 추가
MODEL_SERVER_IP = "192.168.192.111"
MODEL_SERVER_PORT = "8000" # 일단 이 포트로 픽스해두고 문제 생기면 나중에 바꾸는 걸로
MODEL_SERVER_ENDPOINT = "predict" # 임의
MODEL_SERVER_URL = f"http://{MODEL_SERVER_IP}:{MODEL_SERVER_PORT}/{MODEL_SERVER_ENDPOINT}"

## 전처리
def preprocess_image(file_storage):
    try:
        image_data = file_storage.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((256, 256))
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image_array = np.array(image)
        image_array = image_array / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        current_app.logger.error(f"전처리 중 오류: {e}") # print 대신 logger 사용
        return None

## Flask 라우터
@app.route('/')
def home():
    app.logger.info("백엔드 서버 정상적으로 작동 중")
    return "백엔드 서버 작동 중"

@app.route('/api/analyze', methods=['POST'])
def analyze_content():
    # 프론트로부터 요청 왔는지
    if 'file' not in request.files:
        app.logger.warning("요청에 'file' 키가 없음")
        return jsonify({"error": "파일이 없습니다."}), 400
    
    file = request.files['file']

    if file.filename == '':
        app.logger.warning("파일 이름이 비어있음")
        return jsonify({"error": "파일 이름이 없습니다."}), 400

    if not allowed_file(file.filename):
        app.logger.warning(f"허용되지 않는 파일 형식: {file.filename}")
        return jsonify({"error": "허용되지 않는 파일 형식입니다(png, jpg, jpeg만 가능)."}), 400
    
    # 전처리 및 JSON으로 모델 서버에 요청
    try:
        # 원본 파일 -> 256x256 Numpy 배열로 변환
        processed_image = preprocess_image(file)
        if processed_image is None:
             return jsonify({"error": "이미지 파일 형식이 아니거나 손상되었습니다."}), 400

        # Numpy 배열 -> JSON으로 전송 가능한 리스트로 변환
        image_list = processed_image.tolist()
        json_to_send = {'instances': image_list}

        # 모델 서버에 파일이 아닌 'JSON'을 전송
        response_from_model = requests.post(MODEL_SERVER_URL, json = json_to_send)
        
        # 모델 서버의 응답 받음
        model_output_json = response_from_model.json() 
        
        confidence = float(model_output_json['prediction'][0][0])
        is_ai = bool(confidence > 0.5)
        
        final_result = {
            "is_ai_generated": is_ai,
            "confidence": confidence,
            "message": f"{file.filename} 분석 완료"
        }
        app.logger.info(f"파일 분석 성공: {file.filename}, AI 확률: {confidence:.2f}")
        return jsonify(final_result), 200

    except requests.exceptions.ConnectionError as e:
        # 로그 기록 - print 대신 logger.error 사용
        current_app.logger.error(f"모델 서버 연결 실패: {e}")
        return jsonify({"error": "모델 서버에 연결할 수 없습니다. (서버 꺼져 있음)"}), 503 # Service Unavailable
    except Exception as e:
        current_app.logger.error(f"분석 중 오류 발생: {e}")
        return jsonify({"error": "이미지 분석 중 서버 오류가 발생했습니다."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') # 코드 수정 완료하고 ㄱ