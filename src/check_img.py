import os
from PIL import Image

# 데이터셋 경로 (수정해서 사용하세요)
dataset_path = "data/dataset/train"

print(f"🔍 {dataset_path} 폴더의 이미지를 정밀 검사합니다 (Load Test)...")

count = 0
error_count = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file)
            count += 1
            
            # 진행 상황 확인용 (1000장마다 출력)
            if count % 1000 == 0:
                print(f"Checking... {count} images processed")

            try:
                img = Image.open(file_path)
                img.load() # [중요] 실제 픽셀 데이터를 끝까지 읽어들임
            except OSError as e:
                print(f"\n🚨 [범인 발견!] 손상된 파일: {file_path}")
                print(f"   에러 내용: {e}\n")
                error_count += 1
                os.remove(file_path) # 발견 즉시 삭제하려면 주석 해제

print(f"검사 완료. 총 {count}장 중 {error_count}개의 손상된 파일 발견.")