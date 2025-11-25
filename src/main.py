from flask import Flask, jsonify, request, current_app, render_template
## render_template 추가
from flask_cors import CORS
import requests
import io
import os
import logging
import time
import base64

app = Flask(__name__)
CORS(app)

## 로그
file_handler = logging.FileHandler('server.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.propagate = False

## 보안 더블 체크
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # 10MB 용량 제한

def allowed_file(filename):
    """파일 이름에 '.'이 있고, 허용된 확장자인지 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

## 내일 물어보고 정보 추가
MODEL_SERVER_IP = "127.0.0.1" # 수정 필요
MODEL_SERVER_PORT = "8000" # 일단 이 포트로 픽스해두고 문제 생기면 나중에 바꾸는 걸로
MODEL_SERVER_ENDPOINT = "predict"
MODEL_SERVER_URL = f"http://{MODEL_SERVER_IP}:{MODEL_SERVER_PORT}/{MODEL_SERVER_ENDPOINT}"

## Flask 라우터
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# index.html이랑 about.html 추가

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
        app.logger.info(f"가짜 분석 시작: {file.filename}")
        time.sleep(1) # 실제 기다리는 것처럼(테스트용)

        # 무조건 AI라고 우기기(테스트용)
        final_result = {
            "is_ai_generated": True,
            "confidence": 0.985,
            "message": f"{file.filename} 분석 완료 (가짜임)"
        }
        
        app.logger.info(f"가짜 결과 전송 완료: {final_result}")
        return jsonify(final_result), 200
    
        """    
        # 전처리 없이 원본 그대로
        img_bytes = file.read()

        # Base64 문자열로 변환(인코딩)
        base64_string = base64.b64encode(img_bytes).decode('utf-8') 

        # 키 이름 물어보고 바꿔주기
        json_to_send = {
            'filename': file.filename,
            'image_base64': base64_string
        }

        # 모델 서버 전송
        response_from_model = requests.post(MODEL_SERVER_URL, json = json_to_send, timeout = 10)
        
        # 응답 처리 (모델 팀 포맷: filename, prediction, confidence)
        model_output = response_from_model.json() 

        pred_value = model_output.get('prediction') # 결과 라벨 (0 또는 1) / 답 받고 수정
        conf_value = model_output.get('confidence') # 확률

        # 1 = AI, 0 = Real
        is_ai = True if pred_value == 1 else False
        
        final_result = {
            "is_ai_generated": is_ai,
            "confidence": conf_value,
            "message": f"{file.filename} 분석 완료"
        }
        app.logger.info(f"파일 분석 성공: {file.filename}, AI 확률: {conf_value}")
        return jsonify(final_result), 200
        """

    except requests.exceptions.ConnectionError as e:
        current_app.logger.error(f"모델 서버 연결 실패: {e}")
        return jsonify({"error": "모델 서버에 연결할 수 없습니다. (서버 꺼져 있음)"}), 503
    except Exception as e:
        current_app.logger.error(f"분석 중 오류 발생: {e}")
        return jsonify({"error": "이미지 분석 중 서버 오류가 발생했습니다."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050) #포트 5050으로 임의 수정했습니다!(지선)