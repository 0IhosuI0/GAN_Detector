from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

MODEL_SERVER_URL = "http://모델팀_ZeroTier_IP:포트/엔드포인트"
# 엔드포인트는 /predict or /analyze_image

@app.route('/api/analyze', methods=['POST'])
def analyze_content():
    if 'file' not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "파일 이름이 없습니다."}), 400

    # 이전 임시 응답 부분
    try:
        # 모델 서버가 받을 파일 키 'model_file'은 내일 물어보고
        files_to_send = {'model_file': (file.filename, file.stream, file.mimetype)}
        
        # 모델 서버에 HTTP 요청
        response_from_model = requests.post(MODEL_SERVER_URL, files=files_to_send)
        
        # 모델 서버의 응답(JSON)을 받음
        model_output = response_from_model.json() # 예: [0.03, 0.97]

        # 모델 결과를 프론트엔드용 JSON으로 가공
        confidence = float(model_output[1])
        is_ai = bool(confidence > model_output[0])
        
        final_result = {
            "is_ai_generated": is_ai,
            "confidence": confidence,
            "message": f"{file.filename} 분석 완료"
        }
        return jsonify(final_result), 200

    except Exception as e:
        print(f"모델 서버 통신 오류: {e} !!")
        return jsonify({"error": "이미지 분석 중 서버 오류가 발생했습니다."}), 500

if __name__ == '__main__':
    app.run(debug=True)