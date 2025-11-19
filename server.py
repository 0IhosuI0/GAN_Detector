from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__) #flask 애플리케이션 인스턴스 생성
CORS(app)

#라우터 정의
@app.route('/')
def home():
    return "백엔드 서버 작동 중"

 #요청을 처리할 엔드 포인트
@app.route('/api/analyze', methods=['POST'])
def analyze_content():
    #프론트로부터 요청이 왔는지 확인
    if 'file' not in request.files:
        return jsonify({"error: 파일이 요청에 포함되지 않음"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error: 파일 이름이 없음"}), 400
    
    #임시 응답
    dummy_result = {
        "is_ai_generated": True,
        "confidence": 0.97,
        "message": f"{file.filename} 파일 받음. 분석 예정"
    }

    return jsonify(dummy_result), 200

#서버 실행
if __name__ == '__main__':
    app.run(debug=True)