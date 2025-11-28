import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================
# 1. 경로 설정
# ============================
# (1) 특징 벡터 / 라벨 저장된 폴더
DATA_DIR = "/home/nolja30/GAN_Detector/imagenet_ai_0419_sdv4/val"  # <-- 필요하면 수정

# (2) 학습이 끝나고 저장된 모델 폴더
# 예: gendet_saved_models_20251128-040540
MODEL_DIR = "/home/nolja30/GAN_Detector/gendet_saved_models_20251128-044204"  # <-- 여기만 네 폴더 이름으로 수정

FEATURE_FILE = os.path.join(DATA_DIR, "features.npy")
LABEL_FILE   = os.path.join(DATA_DIR, "labels.npy")

BATCH_SIZE = 512  # 평가용 배치 사이즈 (원하면 조절 가능)


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

# ============================
# 2. 데이터 로드
# ============================

def load_dataset():
    print(f"[DATA] Loading features from: {FEATURE_FILE}")
    print(f"[DATA] Loading labels   from: {LABEL_FILE}")

    features_np = np.load(FEATURE_FILE)
    labels_np   = np.load(LABEL_FILE)

    # labels shape: (N, 1) 이라고 가정 → (N,)으로 펼치기
    labels_np = labels_np.reshape(-1).astype(np.float32)

    print(f"[DATA] features shape: {features_np.shape}")
    print(f"[DATA] labels shape  : {labels_np.shape}")
    print(f"[DATA] Real count    : {(labels_np == 0).sum()}")
    print(f"[DATA] Fake count    : {(labels_np == 1).sum()}")
    return features_np, labels_np


# ============================
# 3. 모델 로드
# ============================

def load_models():
    teacher_path    = os.path.join(MODEL_DIR, "teacher.keras")
    student_path    = os.path.join(MODEL_DIR, "student.keras")
    classifier_path = os.path.join(MODEL_DIR, "classifier.keras")

    print(f"[MODEL] Loading teacher    from: {teacher_path}")
    print(f"[MODEL] Loading student    from: {student_path}")
    print(f"[MODEL] Loading classifier from: {classifier_path}")

    custom_objects = {"TransformerBlock": TransformerBlock}

    teacher    = keras.models.load_model(teacher_path, custom_objects=custom_objects, compile=False)
    student    = keras.models.load_model(student_path, custom_objects=custom_objects, compile=False)
    classifier = keras.models.load_model(classifier_path, custom_objects=custom_objects, compile=False)

    return teacher, student, classifier


# ============================
# 4. 간단한 metric 함수들
# ============================

def compute_basic_metrics(probs, labels, threshold=0.5):
    """
    probs : (N,) 확률 (Fake=1일 확률)
    labels: (N,) 실제 라벨 (0: Real, 1: Fake)
    """
    preds = (probs >= threshold).astype(np.float32)

    acc = (preds == labels).mean()

    # confusion matrix
    tp = np.sum((preds == 1) & (labels == 1))  # Fake를 Fake로
    tn = np.sum((preds == 0) & (labels == 0))  # Real을 Real로
    fp = np.sum((preds == 1) & (labels == 0))  # Real을 Fake로
    fn = np.sum((preds == 0) & (labels == 1))  # Fake를 Real로

    # 클래스별 recall 정도
    real_mask = (labels == 0)
    fake_mask = (labels == 1)

    real_acc = np.mean(preds[real_mask] == labels[real_mask]) if real_mask.any() else np.nan
    fake_acc = np.mean(preds[fake_mask] == labels[fake_mask]) if fake_mask.any() else np.nan

    metrics = {
        "accuracy": float(acc),
        "real_acc": float(real_acc),
        "fake_acc": float(fake_acc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    return metrics


def print_metrics(name, metrics):
    print(f"\n=== {name} 성능 ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Real Acc : {metrics['real_acc']:.4f}")
    print(f"Fake Acc : {metrics['fake_acc']:.4f}")
    print("Confusion Matrix (labels: 0=Real, 1=Fake)")
    print(f"  TP(Fake->Fake) : {metrics['tp']}")
    print(f"  TN(Real->Real) : {metrics['tn']}")
    print(f"  FP(Real->Fake) : {metrics['fp']}")
    print(f"  FN(Fake->Real) : {metrics['fn']}")


# ============================
# 5. Teacher 단독 성능 평가
# ============================

def evaluate_teacher(teacher, features_np, labels_np):
    print("\n[Eval] Evaluating Teacher...")

    # Teacher는 sigmoid 출력 (Fake 확률)이라고 가정
    # predict()는 (N, 1) 출력 → (N,)으로 reshape
    probs = teacher.predict(features_np, batch_size=BATCH_SIZE)
    probs = probs.reshape(-1)

    metrics = compute_basic_metrics(probs, labels_np)
    print_metrics("Teacher", metrics)


# ============================
# 6. GenDet(Classifier + discrepancy) 성능 평가
# ============================

def evaluate_gendet(teacher, student, classifier, features_np, labels_np):
    print("\n[Eval] Evaluating GenDet (discrepancy-only classifier)...")

    # 1) Teacher / Student 출력
    z_t = teacher.predict(features_np, batch_size=BATCH_SIZE)  # (N, 1)
    z_s = student.predict(features_np, batch_size=BATCH_SIZE)  # (N, 1)

    # 2) discrepancy = |z_t - z_s|  (input_dim=1)
    abs_diff = np.abs(z_t - z_s)  # (N, 1)

    # 3) Classifier 확률 (Fake 확률)
    probs = classifier.predict(abs_diff, batch_size=BATCH_SIZE)
    probs = probs.reshape(-1)

    metrics = compute_basic_metrics(probs, labels_np)
    print_metrics("GenDet(Classifier)", metrics)

    # 참고용: Real/Fake 별 평균 확률도 출력
    real_mask = labels_np == 0
    fake_mask = labels_np == 1
    if real_mask.any():
        print(f"  [참고] Real 샘플 평균 Fake확률 : {probs[real_mask].mean():.4f}")
    if fake_mask.any():
        print(f"  [참고] Fake 샘플 평균 Fake확률 : {probs[fake_mask].mean():.4f}")


# ============================
# 7. 메인
# ============================

if __name__ == "__main__":
    # GPU 메모리 옵션(원하면 사용)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Enabled memory growth for {len(gpus)} GPU(s).")
        except Exception as e:
            print(f"[WARN] Could not set memory growth: {e}")

    # 1) 데이터 / 모델 로드
    features_np, labels_np = load_dataset()
    teacher, student, classifier = load_models()

    # 2) Teacher 평가
    evaluate_teacher(teacher, features_np, labels_np)

    # 3) GenDet(Classifier + discrepancy) 평가
    evaluate_gendet(teacher, student, classifier, features_np, labels_np)

    print("\n[Done] Evaluation finished.")
