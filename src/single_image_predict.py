import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFCLIPVisionModel, CLIPProcessor
from PIL import Image

# ==========================
# 0. 설정 부분
# ==========================

# 학습이 끝나고 저장된 GenDet 모델 폴더 (1차원 discrepancy 버전)
MODEL_DIR = "/home/nolja30/GAN_Detector/src/dataset"  # <- 네 폴더명으로 수정

# 사용할 CLIP 백본 이름 (학습할 때 썼던 것과 동일)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


# ==========================
# 1. 커스텀 레이어 (TransformerBlock)
# ==========================

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# ==========================
# 2. 모델 / CLIP 로드
# ==========================

def load_gendet_models(model_dir: str):
    teacher_path    = os.path.join(model_dir, "teacher.keras")
    student_path    = os.path.join(model_dir, "student.keras")
    classifier_path = os.path.join(model_dir, "classifier.keras")

    print(f"[MODEL] Loading teacher    from: {teacher_path}")
    print(f"[MODEL] Loading student    from: {student_path}")
    print(f"[MODEL] Loading classifier from: {classifier_path}")

    custom_objects = {"TransformerBlock": TransformerBlock}

    teacher    = keras.models.load_model(teacher_path, custom_objects=custom_objects, compile=False)
    student    = keras.models.load_model(student_path, custom_objects=custom_objects, compile=False)
    classifier = keras.models.load_model(classifier_path, custom_objects=custom_objects, compile=False)

    return teacher, student, classifier


def load_clip(model_name: str):
    print(f"[CLIP] Loading CLIP vision model: {model_name}")
    processor = CLIPProcessor.from_pretrained(model_name)
    vision    = TFCLIPVisionModel.from_pretrained(model_name)
    return processor, vision


# ==========================
# 3. 단일 이미지 전처리 + 특징 추출
# ==========================
IMG_SIZE = 224

def extract_clip_feature(image_path: str,
                         processor: CLIPProcessor,
                         vision_model: TFCLIPVisionModel):
    """
    학습 때와 최대한 동일한 전처리:
    - tf.io.read_file + decode_image
    - tf.image.resize(..., [224,224])
    - 0~1 스케일 후 다시 0~255 uint8로 변환
    - CLIPProcessor(images=...) 호출
    """
    # 1) TF로 로드 + 리사이즈
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0      # [0,1]

    # 2) Numpy로 변환 후 0~255 uint8로
    img_np = img.numpy()                        # (224, 224, 3), float32
    img_uint8 = (img_np * 255).astype(np.uint8)

    # 3) 학습 때와 동일하게 CLIPProcessor 사용
    inputs = processor(images=img_uint8, return_tensors="tf", padding=True)
    pixel_values = inputs["pixel_values"]       # (1, 3, 224, 224)

    # 4) CLIP Vision 모델로 특징 추출
    outputs = vision_model(pixel_values)
    features = outputs.pooler_output            # (1, 768)

    return features


# ==========================
# 4. 단일 이미지 GenDet 추론
# ==========================

def predict_single_image(image_path: str,
                         teacher,
                         student,
                         classifier,
                         processor,
                         vision_model):
    # 1) CLIP 특징 추출
    features = extract_clip_feature(image_path, processor, vision_model)  # (1, 768)

    # 2) Teacher / Student 출력 (둘 다 sigmoid → Fake 확률로 해석)
    z_t = teacher(features, training=False)   # (1, 1)
    z_s = student(features, training=False)   # (1, 1)

    teacher_prob = float(z_t.numpy().reshape(-1)[0])  # Fake 확률
    student_prob = float(z_s.numpy().reshape(-1)[0])

    # 3) discrepancy = |z_t - z_s|  (1차원)
    abs_diff = tf.abs(z_t - z_s)             # (1, 1)

    # 4) Classifier로 최종 Fake 확률
    prob_fake = float(classifier(abs_diff, training=False).numpy().reshape(-1)[0])

    # 5) 최종 판단
    label = "Fake" if prob_fake >= 0.5 else "Real"

    # 6) 부가 정보 출력용
    result = {
        "image_path": image_path,
        "label": label,
        "prob_fake_gendet": prob_fake,
        "prob_fake_teacher": teacher_prob,
        "prob_fake_student": student_prob,
        "discrepancy": float(abs_diff.numpy().reshape(-1)[0]),
    }
    return result


# ==========================
# 5. CLI 진입점
# ==========================

def main():
    parser = argparse.ArgumentParser(description="GenDet: Single Image Real/Fake Prediction")
    parser.add_argument("image", help="Path to image file (jpg/png 등)")
    parser.add_argument("--model_dir", default=MODEL_DIR, help="Saved GenDet model directory")
    parser.add_argument("--clip_model", default=CLIP_MODEL_NAME, help="CLIP model name")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    # GPU 메모리 옵션 (선택)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Enabled memory growth for {len(gpus)} GPU(s).")
        except Exception as e:
            print(f"[WARN] Could not set memory growth: {e}")

    # 모델 / CLIP 로드
    teacher, student, classifier = load_gendet_models(args.model_dir)
    processor, vision_model      = load_clip(args.clip_model)

    # 단일 이미지 추론
    result = predict_single_image(
        image_path=image_path,
        teacher=teacher,
        student=student,
        classifier=classifier,
        processor=processor,
        vision_model=vision_model,
    )

    # 결과 출력
    print("\n================= GenDet Prediction =================")
    print(f"Image          : {result['image_path']}")
    print(f"Final Label    : {result['label']}  "
          f"(Fake prob={result['prob_fake_gendet']:.4f})")
    print("-----------------------------------------------------")
    print(f"Teacher Fake prob : {result['prob_fake_teacher']:.4f}")
    print(f"Student Fake prob : {result['prob_fake_student']:.4f}")
    print(f"|Teacher-Student| : {result['discrepancy']:.6f}")
    print("=====================================================\n")


if __name__ == "__main__":
    main()
