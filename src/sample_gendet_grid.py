import os
import glob
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ============================
# 0. 설정값
# ============================

# (1) val 이미지 폴더 (real/, fake/ 가 들어있는 곳)
VAL_IMAGE_DIR = "/home/nolja30/GAN_Detector/imagenet_ai_0419_sdv4/val"  # <- 네 경로로 수정

# (2) val 특징 벡터 / 라벨 경로
FEATURE_FILE = os.path.join(VAL_IMAGE_DIR, "features.npy")
LABEL_FILE   = os.path.join(VAL_IMAGE_DIR, "labels.npy")

# (3) 사용할 GenDet 모델 폴더 (1차원 discrepancy 버전으로 학습한 폴더)
MODEL_DIR = "/home/nolja30/GAN_Detector/gendet_saved_models_20251128-044204"  # <- 실제 폴더명으로 수정

# (4) 샘플 개수
NUM_ERROR_SAMPLES   = 10   # FP+FN에서 뽑을 개수
NUM_CORRECT_SAMPLES = 10   # TP+TN에서 뽑을 개수

BATCH_SIZE = 512


# ============================
# 1. TransformerBlock 정의 (모델 로드용)
# ============================

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
# 2. 데이터 / 모델 로드
# ============================

def load_dataset():
    print(f"[DATA] Loading features from: {FEATURE_FILE}")
    print(f"[DATA] Loading labels   from: {LABEL_FILE}")

    features_np = np.load(FEATURE_FILE)
    labels_np   = np.load(LABEL_FILE)

    labels_np = labels_np.reshape(-1).astype(np.float32)

    print(f"[DATA] features shape: {features_np.shape}")
    print(f"[DATA] labels shape  : {labels_np.shape}")
    print(f"[DATA] Real count    : {(labels_np == 0).sum()}")
    print(f"[DATA] Fake count    : {(labels_np == 1).sum()}")
    return features_np, labels_np


def load_image_paths():
    pattern = os.path.join(VAL_IMAGE_DIR, "*", "*")
    paths = sorted(glob.glob(pattern))
    print(f"[PATH] Found {len(paths)} image files under {VAL_IMAGE_DIR}")
    return paths


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
# 3. GenDet 예측 & 인덱스 분류
# ============================

def compute_gendet_predictions(teacher, student, classifier, features_np, labels_np):
    print("[Eval] Running GenDet predictions...")

    z_t = teacher.predict(features_np, batch_size=BATCH_SIZE)
    z_s = student.predict(features_np, batch_size=BATCH_SIZE)

    abs_diff = np.abs(z_t - z_s)                     # (N, 1)
    probs = classifier.predict(abs_diff, batch_size=BATCH_SIZE).reshape(-1)  # Fake 확률
    preds = (probs >= 0.5).astype(np.float32)        # 0: Real, 1: Fake

    fp_indices = np.where((preds == 1) & (labels_np == 0))[0]  # Real -> Fake
    fn_indices = np.where((preds == 0) & (labels_np == 1))[0]  # Fake -> Real
    tp_indices = np.where((preds == 1) & (labels_np == 1))[0]  # Fake -> Fake
    tn_indices = np.where((preds == 0) & (labels_np == 0))[0]  # Real -> Real

    print(f"[Eval] FP: {len(fp_indices)}, FN: {len(fn_indices)}, "
          f"TP: {len(tp_indices)}, TN: {len(tn_indices)}")

    return probs, preds, fp_indices, fn_indices, tp_indices, tn_indices


# ============================
# 4. 샘플 인덱스 선택 (랜덤)
# ============================

def pick_samples(fp_indices, fn_indices, tp_indices, tn_indices):
    rng = np.random.default_rng(seed=42)  # 재현 가능하도록 고정 seed

    # FP+FN
    error_indices = np.concatenate([fp_indices, fn_indices])
    if error_indices.size > 0:
        if error_indices.size > NUM_ERROR_SAMPLES:
            error_sample = rng.choice(error_indices, size=NUM_ERROR_SAMPLES, replace=False)
        else:
            error_sample = error_indices
    else:
        print("[WARN] No FP/FN samples found.")
        error_sample = np.array([], dtype=int)

    # TP+TN
    correct_indices = np.concatenate([tp_indices, tn_indices])
    if correct_indices.size > 0:
        if correct_indices.size > NUM_CORRECT_SAMPLES:
            correct_sample = rng.choice(correct_indices, size=NUM_CORRECT_SAMPLES, replace=False)
        else:
            correct_sample = correct_indices
    else:
        print("[WARN] No TP/TN samples found.")
        correct_sample = np.array([], dtype=int)

    print(f"[Sample] Using {len(error_sample)} error samples (FP/FN)")
    print(f"[Sample] Using {len(correct_sample)} correct samples (TP/TN)")

    return error_sample, correct_sample


# ============================
# 5. 그리드 이미지 생성
# ============================

def make_grid_image(out_path, indices, probs, preds, labels_np, image_paths, title):
    if len(indices) == 0:
        print(f"[WARN] No indices for {title}. Skip.")
        return

    n = len(indices)
    cols = min(5, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        img_path = image_paths[idx]
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis("off")

        true_label = int(labels_np[idx])
        pred_label = int(preds[idx])
        prob = float(probs[idx])

        if true_label == 0 and pred_label == 0:
            case = "TN (Real→Real)"
        elif true_label == 0 and pred_label == 1:
            case = "FP (Real→Fake)"
        elif true_label == 1 and pred_label == 1:
            case = "TP (Fake→Fake)"
        else:
            case = "FN (Fake→Real)"

        text = f"{case}\nTrue: {true_label}  Pred: {pred_label}\nFake prob: {prob:.3f}"
        ax.set_title(text, fontsize=9, color="red")

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)  # 제목이랑 겹치지 않게

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PNG] Saved grid image: {out_path}")


# ============================
# 6. 메인
# ============================

if __name__ == "__main__":
    # GPU 메모리 옵션
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Enabled memory growth for {len(gpus)} GPU(s).")
        except Exception as e:
            print(f"[WARN] Could not set memory growth: {e}")

    # 1) 로드
    features_np, labels_np = load_dataset()
    image_paths = load_image_paths()
    assert len(image_paths) == len(labels_np), \
        f"이미지 개수({len(image_paths)})와 라벨 개수({len(labels_np)})가 다릅니다."

    teacher, student, classifier = load_models()

    # 2) 예측 / 분류
    probs, preds, fp_idx, fn_idx, tp_idx, tn_idx = compute_gendet_predictions(
        teacher, student, classifier, features_np, labels_np
    )

    # 3) 샘플 선택
    error_sample, correct_sample = pick_samples(fp_idx, fn_idx, tp_idx, tn_idx)

    # 4) 그리드 PNG 생성
    error_png   = os.path.join(VAL_IMAGE_DIR, "gendet_error_grid1.png")
    correct_png = os.path.join(VAL_IMAGE_DIR, "gendet_correct_grid1.png")

    make_grid_image(
        out_path=error_png,
        indices=error_sample,
        probs=probs,
        preds=preds,
        labels_np=labels_np,
        image_paths=image_paths,
        title="Incorrect Predictions (FP + FN)"
    )

    make_grid_image(
        out_path=correct_png,
        indices=correct_sample,
        probs=probs,
        preds=preds,
        labels_np=labels_np,
        image_paths=image_paths,
        title="Correct Predictions (TP + TN)"
    )

    print("\n[Done] Grid PNGs created.")
