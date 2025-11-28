import os
import numpy as np
import tensorflow as tf
from transformers import TFCLIPVisionModel, CLIPProcessor
from functools import partial

# ===== 1. 경로 설정 =====
VAL_IMAGE_DIR = "/home/nolja30/GAN_Detector/imagenet_ai_0419_sdv4/val"  # val real/fake 폴더가 있는 곳
SAVE_DIR      = VAL_IMAGE_DIR  # features.npy, labels.npy를 이 폴더에 저장

IMG_SIZE   = 224
BATCH_SIZE = 32

FEATURE_FILE = os.path.join(SAVE_DIR, "features.npy")
LABEL_FILE   = os.path.join(SAVE_DIR, "labels.npy")


# ===== 2. 이미지 로더 (너가 쓰던 버전 그대로 재사용) =====
def safe_load_and_decode_image(filepath, img_size):
    """손상된 이미지를 방어하는 로더"""
    try:
        parts = tf.strings.split(filepath, os.path.sep)

        # 경로 뒤에서 두 번째가 레이블(폴더명)이라고 가정
        label_str = parts[-2]

        # 'fake', 'Fake', 'FAKE' 모두 처리
        is_fake = tf.strings.lower(label_str) == 'fake'
        label = tf.cast(is_fake, tf.float32)

        label_vec = tf.reshape(label, [1])

        img_bytes = tf.io.read_file(filepath)
        if tf.strings.length(img_bytes) == 0:
            return tf.zeros([img_size, img_size, 3], dtype=tf.float32), tf.zeros([1], dtype=tf.float32), False

        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0

        img.set_shape([img_size, img_size, 3])
        return img, label_vec, True
    except:
        return tf.zeros([img_size, img_size, 3], dtype=tf.float32), tf.zeros([1], dtype=tf.float32), False


def get_pixel_values_py(image, processor):
    img_np = image.numpy()
    img_uint8 = (img_np * 255).astype(np.uint8)
    inputs = processor(images=img_uint8, return_tensors="tf", padding=True)
    return inputs["pixel_values"]


@tf.function
def run_clip_inference(pixel_values, labels):
    outputs = clip_vision_model(pixel_values)
    features = outputs.pooler_output  # (Batch, 768)
    return features, labels


if __name__ == "__main__":
    # GPU 메모리 옵션(원하면)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Enabled memory growth for {len(gpus)} GPU(s).")
        except Exception as e:
            print(f"[WARN] Could not set memory growth: {e}")

    if os.path.exists(FEATURE_FILE):
        print(f"[SKIP] 이미 {FEATURE_FILE} 가 존재합니다. 새로 추출하지 않습니다.")
        exit(0)

    print("\n[VAL] CLIP feature 추출 시작...")
    print(f"    이미지 폴더 : {VAL_IMAGE_DIR}")
    print(f"    저장 경로   : {SAVE_DIR}")

    # 1) CLIP 모델 로드
    print("[VAL] Loading CLIP model...")
    clip_processor     = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    global clip_vision_model
    clip_vision_model  = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    # 2) 파일 리스트 만들기
    file_pattern = os.path.join(VAL_IMAGE_DIR, "*/*")  # val/real/*.png, val/fake/*.png 같은 구조 가정
    list_ds     = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    total_files = len(tf.io.gfile.glob(file_pattern))
    print(f"[VAL] Total images to process: {total_files}")

    # 3) 이미지 로딩 + 필터링
    dataset = list_ds.map(
        partial(safe_load_and_decode_image, img_size=IMG_SIZE),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.filter(lambda img, label, success: success)
    dataset = dataset.map(lambda img, label, success: (img, label))

    # 4) CLIP processor 적용 (CPU)
    def map_processor_wrapper(image, label):
        pixel_values = tf.py_function(
            func=lambda img: get_pixel_values_py(img, clip_processor),
            inp=[image],
            Tout=tf.float32,
        )
        pixel_values = tf.reshape(pixel_values, [1, 3, IMG_SIZE, IMG_SIZE])
        pixel_values = tf.squeeze(pixel_values, axis=0)
        return pixel_values, label

    processed_ds = dataset.map(map_processor_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    # 5) 배치 + CLIP inference (GPU)
    batched_ds        = processed_ds.batch(BATCH_SIZE)
    feature_ds_batched = batched_ds.map(run_clip_inference, num_parallel_calls=tf.data.AUTOTUNE)
    feature_dataset   = feature_ds_batched.unbatch()

    # 6) 루프 돌면서 numpy로 모으기
    print("[VAL] Extracting features...")
    all_features = []
    all_labels   = []

    try:
        from tqdm import tqdm
        iterator = tqdm(feature_dataset, total=total_files)
    except ImportError:
        print("[VAL] tqdm 미설치. 그냥 진행합니다. (pip install tqdm 추천)")
        iterator = feature_dataset

    for feat, lab in iterator:
        all_features.append(feat.numpy())
        all_labels.append(lab.numpy())

    features_np = np.array(all_features)
    labels_np   = np.array(all_labels)

    # 7) 저장
    print("\n[VAL] Saving features/labels...")
    np.save(FEATURE_FILE, features_np)
    np.save(LABEL_FILE,   labels_np)

    print(f"[VAL] Done! Saved:")
    print(f"  {FEATURE_FILE} (shape: {features_np.shape})")
    print(f"  {LABEL_FILE}   (shape: {labels_np.shape})")