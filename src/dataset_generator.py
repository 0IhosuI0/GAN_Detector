import os
import shutil
import random
from sklearn.model_selection import train_test_split

# --- 설정 ---
# 원본 'real'과 'fake' 이미지가 있는 디렉터리
SOURCE_REAL_DIR = 'data/thumbnails128x128'
SOURCE_FAKE_DIR = 'data/GAN_Generated_Fake_Images'

# 분할된 데이터셋을 저장할 기본 디렉터리
BASE_DIR = 'data/dataset'

# 분할 비율 (훈련, 검증, 테스트)
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15 # 합계는 1.0이어야 합니다.

# --- 스크립트 시작 ---

def create_dir_structure(base_path):
    """필요한 모든 디렉터리를 생성합니다."""
    subdirs = ['train', 'validation', 'test']
    classes = ['real', 'fake']
    for subdir in subdirs:
        for cls in classes:
            os.makedirs(os.path.join(base_path, subdir, cls), exist_ok=True)
    print(f"'{base_path}'에 디렉터리 구조 생성 완료.")

def split_and_copy_files(source_dir, base_path, class_name):
    """파일을 분할하고 해당 디렉터리로 복사합니다."""
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(files)

    # 파일 목록을 비율에 따라 분할
    train_end = int(len(files) * TRAIN_RATIO)
    validation_end = train_end + int(len(files) * VALIDATION_RATIO)

    train_files = files[:train_end]
    validation_files = files[train_end:validation_end]
    test_files = files[validation_end:]

    # 파일 복사
    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(base_path, 'train', class_name, f))
    for f in validation_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(base_path, 'validation', class_name, f))
    for f in test_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(base_path, 'test', class_name, f))

    print(f"'{class_name}' 클래스 파일 복사 완료:")
    print(f"  - 훈련: {len(train_files)}개")
    print(f"  - 검증: {len(validation_files)}개")
    print(f"  - 테스트: {len(test_files)}개")


# 메인 실행 로직
if __name__ == "__main__":
    # 1. 기본 디렉터리 구조 생성
    create_dir_structure(BASE_DIR)

    # 2. 'real' 이미지 분할 및 복사
    print("\n'real' 이미지 처리 중...")
    split_and_copy_files(SOURCE_REAL_DIR, BASE_DIR, 'real')

    # 3. 'fake' 이미지 분할 및 복사
    print("\n'fake' 이미지 처리 중...")
    split_and_copy_files(SOURCE_FAKE_DIR, BASE_DIR, 'fake')

    print("\n모든 작업이 완료되었습니다.")