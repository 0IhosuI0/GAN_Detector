import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import glob
from PIL import Image, ImageOps
from transformers import TFCLIPVisionModel, CLIPProcessor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# =========================================================
# 1. 경로 설정 (본인 환경에 맞게 수정!)
# =========================================================
RESNET_MODEL_PATH = "ResNet_EPOCH100.h5"
GENDET_MODEL_PATH = "gendet/"
TEST_DATA_DIR = "data/dataset/test"  # 0_real, 1_fake 폴더가 있는 곳

# =========================================================
# 2. 모델 로드 (서버 코드와 동일)
# =========================================================
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

print("모델 로딩 중... (서버 없이 로컬에서 실행)")

# GPU 설정
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

# 모델 불러오기
resnet_model = tf.keras.models.load_model(RESNET_MODEL_PATH)
custom_objects = {"TransformerBlock": TransformerBlock}

# 경로 수정 주의 (teacher/student/classifier 각각 로드)
teacher_model = keras.models.load_model(os.path.join(GENDET_MODEL_PATH, "teacher.keras"), custom_objects=custom_objects, compile=False)
student_model = keras.models.load_model(os.path.join(GENDET_MODEL_PATH, "student.keras"), custom_objects=custom_objects, compile=False)
classifier_model = keras.models.load_model(os.path.join(GENDET_MODEL_PATH, "classifier.keras"), custom_objects=custom_objects, compile=False)

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_vision_model = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
print("모든 모델 로딩 완료!")

# =========================================================
# 3. 추론 함수 정의
# =========================================================

def get_resnet_score(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.pad(img, (256, 256), color='black')
        
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = resnet_model.predict(img_array, verbose=0)
        return float(pred[0][0])
    except:
        return 0.5 # 에러시 중립

def get_gendet_score(img_path):
    try:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        
        img_bytes_tf = tf.convert_to_tensor(img_bytes, dtype=tf.string)
        img = tf.image.decode_image(img_bytes_tf, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0
        img_uint8 = (img.numpy() * 255).astype(np.uint8)
        
        inputs = clip_processor(images=img_uint8, return_tensors="tf", padding=True)
        pixel_values = inputs["pixel_values"]
        
        features_numpy = clip_vision_model(pixel_values=pixel_values).pooler_output.numpy()
        
        z_t = teacher_model(features_numpy, training=False)
        z_s = student_model(features_numpy, training=False)
        abs_diff = tf.abs(z_t - z_s)
        score = float(classifier_model(abs_diff, training=False).numpy().reshape(-1)[0])
        return score
    except:
        return 0.5

# =========================================================
# 4. 전체 데이터 평가 실행
# =========================================================

real_paths = glob.glob(os.path.join(TEST_DATA_DIR, "real", "*.*"))
fake_paths = glob.glob(os.path.join(TEST_DATA_DIR, "fake", "*.*"))

print(f"\nReal Images: {len(real_paths)}장")
print(f"Fake Images: {len(fake_paths)}장")

y_true = []
y_pred = []
y_scores = [] # ROC-AUC용 (Fake일 확률)

# 1) Real 평가 (정답 0)
print("Processing Real Images...")
for path in tqdm(real_paths):
    s_res = get_resnet_score(path)
    s_gen = get_gendet_score(path)
    
    # 앙상블: 최대(Max) 방식 사용
    final_score = max(s_res, s_gen)
    
    y_true.append(0)
    y_scores.append(final_score)
    y_pred.append(1 if final_score > 0.5 else 0)

# 2) Fake 평가 (정답 1)
print("Processing Fake Images...")
for path in tqdm(fake_paths):
    s_res = get_resnet_score(path)
    s_gen = get_gendet_score(path)
    
    final_score = (s_res + s_gen) / 2.0
    
    y_true.append(1)
    y_scores.append(final_score)
    y_pred.append(1 if final_score > 0.5 else 0)

# =========================================================
# 5. 결과 리포트 출력
# =========================================================
print("\n" + "="*60)
print("="*60)

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_scores)

print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nDetailed Report:")
print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

# 혼동 행렬 시각화
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred Real', 'Pred Fake'], 
            yticklabels=['Actual Real', 'Actual Fake'])
plt.title(f'Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show() # 로컬 환경이면 창이 뜨고, 서버면 에러날 수 있음 (그땐 plt.savefig 사용)