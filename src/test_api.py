import requests
import base64
import json

# 1. 내 서버 주소 (로컬에서 띄웠으므로 127.0.0.1)
URL = "http://127.0.0.1:35840/predict"
IMAGE_PATH = "KakaoTalk_20251128_144430688_01.jpg" # 테스트할 이미지 파일명

def test_my_server():
    print(f"📂 '{IMAGE_PATH}' 이미지를 준비하는 중...")

    # 2. 이미지를 읽어서 Base64 문자열로 변환
    # (백엔드가 실제로 이렇게 데이터를 가공해서 보낼 겁니다)
    try:
        with open(IMAGE_PATH, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        print("❌ 오류: 이미지 파일이 없습니다. 경로를 확인하세요!")
        return

    # 3. 보낼 데이터 포장 (JSON)
    data = {
        "filename": IMAGE_PATH,
        "image_base64": b64_string
    }

    # 4. 서버로 전송 (POST 요청)
    print("🚀 서버로 전송 중...")
    try:
        response = requests.post(URL, json=data)
        
        # 5. 결과 확인
        if response.status_code == 200:
            print("\n✅ [성공] 서버로부터 응답이 왔습니다!")
            print("-------------------------------------")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            print("-------------------------------------")
        else:
            print(f"🔥 [실패] 서버 에러 발생: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("❌ [오류] 서버가 꺼져있습니다. ai_server.py를 먼저 실행하세요!")

if __name__ == "__main__":
    test_my_server()