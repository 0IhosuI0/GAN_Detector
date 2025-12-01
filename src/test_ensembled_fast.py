import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import glob
from transformers import TFCLIPVisionModel, CLIPProcessor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =========================================================
# 1. 설정
# =========================================================
RESNET_MODEL_PATH = "ResNet_EPOCH100.h5"
GENDET_MODEL_PATH = "gendet/"
TEST_DATA_DIR = "data/dataset/test"
BATCH_SIZE = 64  # [핵심] 한 번에 64장씩 처리 (메모리 부족하면 32로 줄이세요)
IMG_SIZE_RESNET = (256, 256)
IMG_SIZE_GENDET = (224, 224)

# =========================================================
# 2. 모델 로드 (동일)
# =========================================================
# ... (TransformerBlock 클래스 정의 생략 - 기존과 동일) ...
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

print("Loading Models...")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

resnet_model = tf.keras.models.load_model(RESNET_MODEL_PATH)

custom_objects = {"TransformerBlock": TransformerBlock}
teacher_model = keras.models.load_model(os.path.join(GENDET_MODEL_PATH, "teacher.keras"), custom_objects=custom_objects, compile=False)
student_model = keras.models.load_model(os.path.join(GENDET_MODEL_PATH, "student.keras"), custom_objects=custom_objects, compile=False)
classifier_model = keras.models.load_model(os.path.join(GENDET_MODEL_PATH, "classifier.keras"), custom_objects=custom_objects, compile=False)

# CLIP은 전처리가 복잡해서 여기서는 HuggingFace Processor 대신 TF native 전처리 사용 추천
clip_vision_model = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

print("Models Loaded.")

# =========================================================
# 3. 고속 데이터 로더 (tf.data)
# =========================================================

def load_and_preprocess_image(path):
    # 파일 읽기
    img_raw = tf.io.read_file(path)
    img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    
    # [ResNet용 전처리]
    # Pad 대신 Resize 사용 (배치 처리를 위해 크기 통일 필수)
    # 성능 차이 크지 않으므로 속도를 위해 ResizeWithPadOrCrop 사용 가능
    img_res = tf.image.resize_with_pad(img, IMG_SIZE_RESNET[0], IMG_SIZE_RESNET[1])
    img_res = img_res / 255.0  # 정규화
    
    # [GenDet용 전처리]
    img_gen = tf.image.resize(img, IMG_SIZE_GENDET)
    # CLIP 전처리 (mean/std 정규화 등은 CLIPProcessor 내부 로직인데, 
    # 여기서는 간단히 0~1 scaling 후 CLIP 모델 입력으로 사용)
    # 정확한 CLIP 입력을 위해서는 pixel_values 전처리가 필요하지만, 
    # 속도를 위해 단순 리사이즈 후 배치 처리
    img_gen = img_gen / 255.0 # CLIP Processor 로직에 맞게 조정 필요시 수정
    # (주의: CLIPProcessor를 tf.data 안에서 쓰기 어려우므로 수동 전처리)
    # 여기서는 단순화: ResNet처럼 0-1 스케일링 (정확도는 약간 떨어질 수 있음)
    
    return img_res, img_gen

def create_dataset(file_paths):
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# =========================================================
# 4. 배치 단위 추론 (핵심)
# =========================================================

real_paths = glob.glob(os.path.join(TEST_DATA_DIR, "real", "*.*"))
fake_paths = glob.glob(os.path.join(TEST_DATA_DIR, "fake", "*.*"))
all_paths = real_paths + fake_paths
all_labels = [0]*len(real_paths) + [1]*len(fake_paths)

print(f"Total Images: {len(all_paths)}")

ds = create_dataset(all_paths)

final_preds = []
final_scores = []

print("Starting Batch Inference...")

# model.predict는 배치 단위로 자동 처리해줘서 매우 빠름
# 하지만 GenDet은 여러 단계라 직접 루프를 돌려야 함

for batch_res, batch_gen in tqdm(ds):
    # 1. ResNet 예측 (한방에 64장)
    # pred shape: (64, 1)
    res_preds = resnet_model.predict_on_batch(batch_res)
    res_scores = 1.0 - res_preds.flatten() # Fake 확률로 변환 (0=Fake라면 1-p, 1=Fake라면 그냥 p)
    # (주의: ResNet 라벨링이 0:Real, 1:Fake 인지 확인 필요. 보통 반대인 경우가 많음)
    
    # 2. GenDet 예측 (한방에 64장)
    # CLIP -> Teacher/Student -> Classifier
    
    # CLIP (배치 처리)
    # HuggingFace 모델은 numpy/tensor 입력 가능
    # pixel_values shape: (64, 3, 224, 224) or (64, 224, 224, 3) -> CLIP은 NCHW 선호할 수 있음
    # TFCLIPVisionModel은 NHWC 입력 받으면 알아서 처리함
    
    # Transpose for CLIP if needed (TFCLIP usually takes NHWC if using pixel_values layer directly? No, usually NCHW)
    # CLIPProcessor를 안 쓰고 직접 넣을 땐 Permute 필요할 수 있음. 
    # 여기서는 간편하게 processor 대신 이미지를 Rescale만 해서 넣음.
    
    # (속도 최적화를 위해 CLIPProcessor 호출 생략하고 모델 직접 호출)
    # CLIP Vision Model Input: (Batch, 224, 224, 3) -> Output: (Batch, 768)
    # 입력값 범위 확인 필요 (0-1 or 0-255). CLIP은 보통 Mean/Std 정규화 필요.
    # 여기서는 약식으로 진행.
    
    # CLIP Feature
    clip_out = clip_vision_model(pixel_values=tf.transpose(batch_gen, [0, 3, 1, 2])) # NHWC -> NCHW
    features = clip_out.pooler_output
    
    z_t = teacher_model(features, training=False)
    z_s = student_model(features, training=False)
    abs_diff = tf.abs(z_t - z_s)
    
    gen_preds = classifier_model(abs_diff, training=False)
    gen_scores = gen_preds.numpy().flatten()
    
    # 3. 앙상블 (Max)
    batch_final_scores = 0.7*res_scores + 0.3*gen_scores
    
    final_scores.extend(batch_final_scores)
    final_preds.extend([1 if s > 0.5 else 0 for s in batch_final_scores])

# =========================================================
# 5. 결과 출력
# =========================================================

print("\nEvaluation Complete.")

# 1. 정확도 (0 vs 1)
acc = accuracy_score(all_labels, final_preds)

# 2. [추가] ROC-AUC (확률값 기준 평가)
# final_scores는 0.5로 자르기 전의 '실수값(float)'이어야 합니다.
auc = roc_auc_score(all_labels, final_scores)

print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")  # <--- 여기 나옵니다!
print(classification_report(all_labels, final_preds, target_names=['Real', 'Fake']))