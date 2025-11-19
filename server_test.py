from flask import Flask, jsonify, request, current_app
from flask_cors import CORS
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
import logging

app = Flask(__name__)
CORS(app)

# -----------------------
# 로그 설정
# -----------------------
file_handler = logging.FileHandler('server.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.propagate = False

# -----------------------
# 파일 제한
# -----------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB 제한


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------
# 이미지 전처리
# -----------------------
def preprocess_image(file_storage):
    try:
        image_data = file_storage.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((256, 256))

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image_array = np.array(image).astype('float32') / 255.0
        return np.expand_dims(image_array, axis=0)

    except UnidentifiedImageError:
        current_app.logger.error("이미지 파일이 아님 또는 손상됨")
        return None
    except Exception as e:
        current_app.logger.error(f"전처리 중 오류: {e}")
        return None


# -----------------------
# 라우터
# -----------------------
@app.route('/')
def home():
    app.logger.info("백엔드 서버 정상 작동 중")
    return "백엔드 서버 작동 중"


# -----------------------
# 분석 API (테스트 모드)
# -----------------------
@app.route('/api/analyze', methods=['POST'])
def analyze_content():

    if 'file' not in request.files:
        app.logger.warning("'file' 키 없음")
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files['file']

    if file.filename == '':
        app.logger.warning("파일 이름 없음")
        return jsonify({"error": "파일 이름이 없습니다."}), 400

    if not allowed_file(file.filename):
        app.logger.warning(f"허용되지 않는 확장자: {file.filename}")
        return jsonify({"error": "허용되지 않는 파일 형식입니다(png, jpg, jpeg만 가능)."}), 400

    # 전처리
    processed_image = preprocess_image(file)
    if processed_image is None:
        return jsonify({"error": "이미지 파일 형식이 아니거나 손상되었습니다."}), 400

    # -----------------------
    # 테스트 모드: 모델 서버 호출 없음
    # -----------------------
    final_result = {
        "is_ai_generated": False,
        "confidence": 0.0,
        "message": f"{file.filename} 분석 완료 (테스트 모드: 모델 서버 없음)"
    }

    app.logger.info(f"파일 분석 성공(테스트 모드): {file.filename}")
    return jsonify(final_result), 200


# -----------------------
# 실행
# -----------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
