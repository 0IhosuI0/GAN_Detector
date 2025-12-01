from flask import Flask, jsonify, request, current_app, render_template
from flask_cors import CORS
import requests
import io
import os
import logging
import time
import base64
import json

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

MODEL_SERVER_IP = "192.168.192.111" # 수정함
MODEL_SERVER_PORT = "35840" # 포트 수정함
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
        app.logger.warning("파일이 비어있음")
        return jsonify({"error": "파일이 없습니다."}), 400

    if not allowed_file(file.filename):
        app.logger.warning(f"허용되지 않는 파일 형식: {file.filename}")
        return jsonify({"error": "허용되지 않는 파일 형식입니다(png, jpg, jpeg만 가능)."}), 400
    
    # 전처리 및 JSON으로 모델 서버에 요청
    try:
        """
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
        response_from_model = requests.post(MODEL_SERVER_URL, json = json_to_send, timeout = 60) # timeout 60초로 변경함
        
        # 모델 서버 상태 코드 확인 - 200 아님 에러
        if response_from_model.status_code != 200:
            app.logger.error(f"모델 서버 에러 반환: {response_from_model.text}")
            return jsonify({
                "error": "AI 모델 서버 내부 오류",
                "detail": response_from_model.text
            }), 502

        # 응답 처리 (모델 팀 포맷: filename, prediction, confidence)
        model_output = response_from_model.json() 

        pred_value = model_output.get('prediction') # 결과 라벨 (0 또는 1) / 답 받고 수정
        conf_value = model_output.get('confidence') # 확률

        # 방어 코드 추가 - 연결 오류 떴을 때
        if pred_value is None or conf_value is None:
            app.logger.error(f"모델 응답 데이터 누락: {model_output}")
            return jsonify({"error": "AI 모델이 분석에 실패했습니다. (결과값 None)"}), 500

        # 1 = AI, 0 = Real
        is_ai = True if pred_value == 1 else False
        
        final_result = {
            "is_ai_generated": is_ai,
            "confidence": conf_value,
            "message": f"{file.filename} 분석 완료"
        }
        app.logger.info(f"파일 분석 성공: {file.filename}, AI 확률: {conf_value}")
        return jsonify(final_result), 200

    except requests.exceptions.Timeout:
        current_app.logger.error("모델 서버 응답 시간 초과 (Timeout)")
        return jsonify({"error": "모델 분석 시간이 초과되었습니다.(60초)"}), 504

    except requests.exceptions.ConnectionError:
        current_app.logger.error("모델 서버 연결 실패 (로딩 가능성 높음)")
        return jsonify({"error": "모델 서버가 로딩(실행) 중입니다. 잠시 후 다시 시도해 주세요."}), 503

    except json.JSONDecodeError:
        current_app.logger.error("모델 응답 JSON 파싱 실패")
        return jsonify({"error": "모델 서버 응답 형식을 확인해 주세요."}), 500

    except Exception as e:
        current_app.logger.error(f"알 수 없는 서버 오류: {e}")
        return jsonify({"error": f"서버 내부 오류가 발생했습니다. ({str(e)})"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)