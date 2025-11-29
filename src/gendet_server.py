import os
import io
import base64
from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TFCLIPVisionModel, CLIPProcessor
from PIL import Image

# ==========================
# 0. 설정
# ==========================

# GenDet 모델이 저장된 폴더 (네 환경에 맞게 수정!)
MODEL_DIR = "/home/nolja30/GAN_Detector/src/dataset"

TEACHER_PATH = os.path.join(MODEL_DIR, "teacher.keras")
STUDENT_PATH = os.path.join(MODEL_DIR, "student.keras")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.keras")

# 학습 때 사용한 CLIP 백본
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# 이미지 전처리 사이즈 (학습 때와 동일)
IMG_SIZE = 224

# ==========================
# 1. Pydantic Request 모델
# ==========================

class ImageRequest(BaseModel):
    filename: str
    image_base64: str


# ==========================
# 2. 커스텀 레이어 정의 (TransformerBlock)
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
# 3. 전역 객체 (모델/CLIP) 로드
# ==========================

app = FastAPI()

teacher_model: Optional[keras.Model] = None
student_model: Optional[keras.Model] = None
classifier_model: Optional[keras.Model] = None
clip_processor: Optional[CLIPProcessor] = None
clip_vision_model: Optional[TFCLIPVisionModel] = None


def setup_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Enabled memory growth for {len(gpus)} GPU(s).")
        except Exception as e:
            print(f"[WARN] Could not set memory growth: {e}")


def load_gendet_models():
    global teacher_model, student_model, classifier_model

    custom_objects = {"TransformerBlock": TransformerBlock}

    print("[MODEL] Loading GenDet models from:", MODEL_DIR)
    teacher_model = keras.models.load_model(
        TEACHER_PATH, custom_objects=custom_objects, compile=False
    )
    student_model = keras.models.load_model(
        STUDENT_PATH, custom_objects=custom_objects, compile=False
    )
    classifier_model = keras.models.load_model(
        CLASSIFIER_PATH, custom_objects=custom_objects, compile=False
    )

    print("[MODEL] Teacher, Student, Classifier loaded successfully.")


def load_clip_backbone():
    global clip_processor, clip_vision_model

    print(f"[CLIP] Loading CLIP backbone: {CLIP_MODEL_NAME}")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_vision_model = TFCLIPVisionModel.from_pretrained(CLIP_MODEL_NAME)
    print("[CLIP] CLIP Processor & Vision model loaded.")


@app.on_event("startup")
def startup_event():
    """서버 시작 시 한 번만 실행."""
    setup_gpu_memory_growth()
    load_gendet_models()
    load_clip_backbone()
    print("[SERVER] GenDet FastAPI server is ready.")


# ==========================
# 4. 전처리 & 특징 추출 함수
# ==========================

def preprocess_image_from_base64(b64_string: str) -> tf.Tensor:
    """
    - base64 문자열 → RGB 이미지 디코드
    - 학습 때와 동일하게: decode_image → resize(224,224) → [0,1] → uint8
    - CLIPProcessor에 넣기 위한 pixel_values 텐서 반환
    """
    # 1) base64 디코드 → tf.Tensor of bytes
    image_bytes = base64.b64decode(b64_string)
    img_bytes_tf = tf.convert_to_tensor(image_bytes, dtype=tf.string)

    # 2) TF로 이미지 디코드
    img = tf.image.decode_image(img_bytes_tf, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0  # [0,1]

    # 3) numpy → uint8 (학습 때와 동일)
    img_np = img.numpy()  # (224,224,3)
    img_uint8 = (img_np * 255).astype(np.uint8)

    # 4) CLIPProcessor에 넣기
    inputs = clip_processor(images=img_uint8, return_tensors="tf", padding=True)
    pixel_values = inputs["pixel_values"]  # (1,3,224,224)

    return pixel_values


def extract_clip_feature(pixel_values: tf.Tensor) -> tf.Tensor:
    """
    pixel_values: (1,3,224,224)
    반환: features (1,768)
    """
    outputs = clip_vision_model(pixel_values)
    features = outputs.pooler_output  # (1,768)
    return features


# ==========================
# 5. GenDet 추론 함수
# ==========================

def gendet_predict_from_feature(features: tf.Tensor):
    """
    features: (1,768) CLIP feature
    반환: dict (teacher_prob, student_prob, discrepancy, gendet_prob, label)
    """
    # Teacher / Student 출력 (sigmoid로 Fake 확률)
    z_t = teacher_model(features, training=False)   # (1,1)
    z_s = student_model(features, training=False)   # (1,1)

    teacher_prob = float(z_t.numpy().reshape(-1)[0])  # Fake 확률
    student_prob = float(z_s.numpy().reshape(-1)[0])

    # discrepancy = |z_t - z_s|  (1차원)
    abs_diff = tf.abs(z_t - z_s)                    # (1,1)
    discrepancy_val = float(abs_diff.numpy().reshape(-1)[0])

    # Classifier로 최종 Fake 확률
    gendet_prob = float(
        classifier_model(abs_diff, training=False).numpy().reshape(-1)[0]
    )

    # 최종 라벨 (임계값 0.5 기준)
    label = "Fake" if gendet_prob >= 0.5 else "Real"

    return {
        "teacher_fake_prob": teacher_prob,
        "student_fake_prob": student_prob,
        "discrepancy": discrepancy_val,
        "gendet_fake_prob": gendet_prob,
        "label": label,
    }


# ==========================
# 6. FastAPI 엔드포인트
# ==========================

@app.post("/predict")
def predict_image(request: ImageRequest):
    if teacher_model is None or student_model is None or classifier_model is None:
        return {"error": "GenDet models are not loaded."}
    if clip_processor is None or clip_vision_model is None:
        return {"error": "CLIP backbone is not loaded."}

    try:
        # 1) base64 → pixel_values (학습 때와 동일 전처리)
        pixel_values = preprocess_image_from_base64(request.image_base64)

        # 2) CLIP feature 추출
        features = extract_clip_feature(pixel_values)

        # 3) GenDet 추론
        result = gendet_predict_from_feature(features)

        # 4) 응답 JSON 구성
        response = {
            "filename": request.filename,
            "final_label": result["label"],
            "gendet_fake_prob": round(result["gendet_fake_prob"], 6),
            "teacher_fake_prob": round(result["teacher_fake_prob"], 6),
            "student_fake_prob": round(result["student_fake_prob"], 6),
            "discrepancy": round(result["discrepancy"], 6),
        }
        return response

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


# ==========================
# 7. 로컬 실행 (uvicorn)
# ==========================

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 으로 열어서 외부(같은 LAN)에서도 접근 가능하게 할 수도 있음
    uvicorn.run("gendet_server:app", host="0.0.0.0", port=35840, reload=False)
