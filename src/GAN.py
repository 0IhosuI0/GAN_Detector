# GenDet: Towards Good Generalizations for AI-Generated Image Detection
# 논문 (arXiv:2312.08880v1) 구현 - TensorFlow/Keras
# *** v5 Final: 모델 충돌 해결 & 배치 추론 적용 (가장 빠르고 안정적인 버전) ***

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses
from transformers import TFCLIPVisionModel, CLIPProcessor
import numpy as np
import os
from functools import partial

# --- 1. 모델 아키텍처 정의 (변경 없음) ---
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

def create_gendet_component(input_dim=768, model_dim=196, num_heads=4, ff_dim=784, name="component"):
    inputs = keras.Input(shape=(input_dim,), name=f"{name}_input")
    x = layers.Dense(model_dim, name=f"{name}_projection")(inputs)
    x = layers.Reshape((1, model_dim))(x)
    x = TransformerBlock(model_dim, num_heads, ff_dim, name=f"{name}_transformer")(x)
    x = layers.Reshape((model_dim,))(x)
    #outputs = layers.Dense(1, name=f"{name}_output")(x)
    outputs = layers.Dense(1, activation="sigmoid", name=f"{name}_output")(x)
    return keras.Model(inputs, outputs, name=name)

def create_augmenter(input_dim=768, model_dim=196, num_heads=4, ff_dim=784, name="augmenter"):
    inputs = keras.Input(shape=(input_dim,), name=f"{name}_input")
    x = layers.Dense(model_dim, name=f"{name}_projection")(inputs)
    x = layers.Reshape((1, model_dim))(x)
    x = TransformerBlock(model_dim, num_heads, ff_dim, name=f"{name}_transformer")(x)
    x = layers.Reshape((model_dim,))(x)
    outputs = layers.Dense(input_dim, name=f"{name}_output")(x)
    return keras.Model(inputs, outputs, name=name)

def create_classifier(input_dim=1, model_dim=128, num_heads=4, ff_dim=512, name="classifier"):
    inputs = keras.Input(shape=(input_dim,), name=f"{name}_input")
    x = layers.Dense(model_dim, name=f"{name}_projection")(inputs)
    x = layers.Reshape((1, model_dim))(x)
    x = TransformerBlock(model_dim, num_heads, ff_dim, name=f"{name}_transformer")(x)
    x = layers.Reshape((model_dim,))(x)
    outputs = layers.Dense(1, activation='sigmoid', name=f"{name}_output")(x)
    return keras.Model(inputs, outputs, name=name)

#def create_classifier(input_dim=1, model_dim=128, num_heads=4, ff_dim=512, name="classifier"):
#    inputs = keras.Input(shape=(input_dim,), name=f"{name}_input")
#    x = layers.Dense(model_dim, name=f"{name}_projection")(inputs)
#    x = layers.Reshape((1, model_dim))(x)
#    x = TransformerBlock(model_dim, num_heads, ff_dim, name=f"{name}_transformer")(x)
#    x = layers.Reshape((model_dim,))(x)
#    outputs = layers.Dense(1, activation='sigmoid', name=f"{name}_output")(x)
#    return keras.Model(inputs, outputs, name=name)


# --- 2. GenDet 메인 모델 및 훈련 로직 (변경 없음) ---
class GenDet(keras.Model):
    def __init__(self, teacher, student, augmenter, classifier, margin=2.0, **kwargs):
        super(GenDet, self).__init__(**kwargs)
        self.teacher = teacher
        self.student = student
        self.augmenter = augmenter
        self.classifier = classifier
        self.margin = margin

        self.real_dist_tracker = keras.metrics.Mean(name="real_dist")
        self.fake_dist_tracker = keras.metrics.Mean(name="fake_dist")
        self.loss_a_tracker = keras.metrics.Mean(name="loss_a")
        self.loss_b_tracker = keras.metrics.Mean(name="loss_b")

    def compile(self, t_optimizer, s_optimizer, a_optimizer, c_optimizer):
        super(GenDet, self).compile()
        self.t_optimizer = t_optimizer
        self.s_optimizer = s_optimizer
        self.a_optimizer = a_optimizer
        self.c_optimizer = c_optimizer
        self.bce_loss = losses.BinaryCrossentropy()
        self.mse_loss = losses.MeanSquaredError()

    def pretrain_teacher(self, dataset, epochs=10):
        print("\n--- [Step 0] Pre-training Teacher Network ---")
        self.teacher.compile(optimizer=self.t_optimizer, loss=self.bce_loss, metrics=['accuracy'])
        self.teacher.fit(dataset, epochs=epochs)
        print("--- Teacher pre-training finished ---")

    def train_step(self, data):
        real_features, fake_features = data

        for _ in range(3):
            with tf.GradientTape() as tape:
                z_t_r = self.teacher(real_features, training=False)
                z_s_r = self.student(real_features, training=True)
                loss_a = self.mse_loss(z_t_r, z_s_r)  # Real Loss

                aug_fake_features = self.augmenter(fake_features, training=False)
                z_t_f = self.teacher(aug_fake_features, training=False)
                z_s_f = self.student(aug_fake_features, training=True)

                #  l2_normalize 제거, L2 거리 기반 margin
                dist = tf.norm(z_t_f - z_s_f, axis=-1)  # shape: (batch,)
                loss_b = tf.reduce_mean(tf.maximum(0.0, self.margin - dist))

                student_loss = loss_a + (5.0 * loss_b)

            grads = tape.gradient(student_loss, self.student.trainable_variables)
            self.s_optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        # === 2) Augmenter 업데이트 ===
        with tf.GradientTape() as tape:
            aug_fake_features = self.augmenter(fake_features, training=True)
            z_t_f = self.teacher(aug_fake_features, training=False)
            z_s_f = self.student(aug_fake_features, training=False)
            augmenter_loss = self.mse_loss(z_t_f, z_s_f)

        grads = tape.gradient(augmenter_loss, self.augmenter.trainable_variables)
        self.a_optimizer.apply_gradients(zip(grads, self.augmenter.trainable_variables))

        # === 3) 모니터링용 거리 계산 ===
        aug_fake_features = self.augmenter(fake_features, training=False)
        z_t_r = self.teacher(real_features, training=False)
        z_s_r = self.student(real_features, training=False)
        z_t_f = self.teacher(aug_fake_features, training=False)
        z_s_f = self.student(aug_fake_features, training=False)

        real_dist = tf.reduce_mean(tf.abs(z_t_r - z_s_r))
        fake_dist = tf.reduce_mean(tf.abs(z_t_f - z_s_f))

        #  add_metric 대신 metric 객체에 update_state
        self.real_dist_tracker.update_state(real_dist)
        self.fake_dist_tracker.update_state(fake_dist)
        self.loss_a_tracker.update_state(loss_a)
        self.loss_b_tracker.update_state(loss_b)

        # fit() 로그에는 return 딕셔너리 + metrics 모두 같이 출력됨
        return {
            "student_loss": student_loss,
            "augmenter_loss": augmenter_loss,
        }

    def train_classifier(self, dataset, epochs=20):
        print("\n--- [Step 4] Training Final Classifier ---")
        self.classifier.compile(optimizer=self.c_optimizer, loss=self.bce_loss, metrics=['accuracy'])
        
        @tf.function
        def create_discrepancy_dataset(features, labels):
            z_t = self.teacher(features, training=False)
            z_s = self.student(features, training=False)
            diff = z_t - z_s
            abs_diff = tf.abs(diff)
            
            # [제안] teacher, student, abs_diff를 한 번에 사용
            #discrepancy = tf.concat([z_t, z_s, abs_diff], axis=-1)  # shape: (batch, 3)
            discrepancy = abs_diff
            return discrepancy, labels

        classifier_ds = dataset.map(create_discrepancy_dataset)
        self.classifier.fit(classifier_ds, epochs=epochs)
        print("--- Classifier training finished ---")

    def predict_feature(self, features):
        # features: (batch, 768) 라고 가정
        z_t = self.teacher(features, training=False)
        z_s = self.student(features, training=False)
        diff = z_t - z_s
        abs_diff = tf.abs(diff)
        discrepancy = tf.concat([z_t, z_s, abs_diff], axis=-1)  # (batch, 3)

        probability = self.classifier(discrepancy, training=False)  # (batch, 1)

        prob_np = probability.numpy().reshape(-1)  # 1D로
        label = "Fake" if prob_np[0] > 0.5 else "Real"
        return label, float(prob_np[0])


# --- 3. 데이터 파이프라인 (안전한 로딩 + 분리된 CLIP 추론) ---

def safe_load_and_decode_image(filepath, img_size):
    """손상된 이미지를 방어하는 로더"""
    try:
        parts = tf.strings.split(filepath, os.path.sep)
        
        # [수정됨] 폴더명 추출 및 대소문자 처리 강화
        # 경로 뒤에서 두 번째가 레이블(폴더명)이라고 가정
        label_str = parts[-2]
        
        # 'fake', 'Fake', 'FAKE' 모두 처리하기 위해 소문자로 변환 후 비교
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

# [CPU 전용] CLIP Processor 실행 함수
def get_pixel_values_py(image, processor):
    # 1. 입력받은 텐서를 Numpy 배열로 변환 (중요!)
    img_np = image.numpy() 
    
    # 2. 0-1 실수형을 0-255 정수형(uint8)으로 변환
    # 이제 img_np는 numpy 배열이므로 .astype() 사용 가능
    img_uint8 = (img_np * 255).astype(np.uint8)
    
    # 3. Processor 호출
    inputs = processor(images=img_uint8, return_tensors="tf", padding=True)
    return inputs['pixel_values']

# [GPU 전용] CLIP Model 실행 함수 (Graph Mode)
@tf.function
def run_clip_inference(pixel_values, labels):
    # pixel_values: (Batch, 3, 224, 224) 혹은 설정에 따라 (Batch, 224, 224, 3)
    outputs = clip_vision_model(pixel_values)
    features = outputs.pooler_output # (Batch, 768)
    return features, labels


if __name__ == '__main__':
    # --- 하이퍼파라미터 설정 ---
    BATCH_SIZE = 32
    IMG_SIZE = 224
    FEATURE_DIM = 768
    SAVE_DIR = "/home/nolja30/GAN_Detector/src/dataset" # 저장할 폴더
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    # [경로 설정] 실제 데이터셋 경로
    train_data_dir = "/home/nolja30/GAN_Detector/imagenet_ai_0419_sdv4/train"

    # =========================================================
    # [Part 1] 특징 추출 및 저장 (파일이 없을 때만 실행)
    # =========================================================
    feature_file = os.path.join(SAVE_DIR, 'features.npy')
    label_file = os.path.join(SAVE_DIR, 'labels.npy')

    if not os.path.exists(feature_file):
        print("\n [Step 1] 저장된 파일이 없어 특징 추출을 시작합니다...")
        print("   (이 과정은 1회만 수행되며, 시간이 소요됩니다.)")

        # --- CLIP 모델 로드 ---
        print("Loading CLIP model...")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_vision_model = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # --- 데이터셋 파이프라인 구축 ---
        file_pattern = os.path.join(train_data_dir, '*/*')
        list_ds = tf.data.Dataset.list_files(file_pattern, shuffle=False) # 추출할 땐 셔플 불필요
        TOTAL_SAMPLES = len(tf.io.gfile.glob(file_pattern))
        print(f"Total samples to process: {TOTAL_SAMPLES}")

        # 1. 로드 & 필터링
        dataset = list_ds.map(partial(safe_load_and_decode_image, img_size=IMG_SIZE), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.filter(lambda img, label, success: success)
        dataset = dataset.map(lambda img, label, success: (img, label))

        # 2. 전처리 (CPU)
        def map_processor_wrapper(image, label):
            pixel_values = tf.py_function(
                func=lambda img: get_pixel_values_py(img, clip_processor),
                inp=[image], Tout=tf.float32
            )
            pixel_values = tf.reshape(pixel_values, [1, 3, IMG_SIZE, IMG_SIZE])
            pixel_values = tf.squeeze(pixel_values, axis=0)
            return pixel_values, label

        processed_ds = dataset.map(map_processor_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

        # 3. 배치 및 추론 (GPU)
        batched_ds = processed_ds.batch(BATCH_SIZE)
        feature_ds_batched = batched_ds.map(run_clip_inference, num_parallel_calls=tf.data.AUTOTUNE)
        feature_dataset = feature_ds_batched.unbatch()

        # 4. ★★★ [수정됨] 실제 추출 및 리스트 저장 루프 ★★★
        print("Extracting features (This may take a while)...")
        all_features = []
        all_labels = []

        # 진행률 표시를 위해 tqdm 사용 (없으면 pip install tqdm)
        try:
            from tqdm import tqdm
            iterator = tqdm(feature_dataset, total=TOTAL_SAMPLES)
        except ImportError:
            print("tqdm not found. installing recommended: pip install tqdm")
            iterator = feature_dataset

        for features, labels in iterator:
            all_features.append(features.numpy())
            all_labels.append(labels.numpy())

        # 5. 파일 저장
        print("\n Saving to disk...")
        features_np = np.array(all_features)
        labels_np = np.array(all_labels)

        np.save(feature_file, features_np)
        np.save(label_file, labels_np)
        print(" Extraction & Saving Complete!")

    else:
        print("\n [Step 1] 저장된 특징 파일을 발견했습니다. 추출 과정을 건너뜁니다.")

    # =========================================================
    # [Part 2] 고속 데이터 로딩 및 학습
    # =========================================================
    print(f"\n📂 Loading data from: {SAVE_DIR}")
    features_np = np.load(feature_file)
    labels_np = np.load(label_file)
    
    TOTAL_SAMPLES = features_np.shape[0]
    print(f"✅ Data Loaded! Count: {TOTAL_SAMPLES}, Shape: {features_np.shape}")

    # ★★★ [중요] 메모리에 있는 Numpy 배열로 고속 데이터셋 생성
    fast_dataset = tf.data.Dataset.from_tensor_slices((features_np, labels_np))
    
    # 셔플 및 배치
    fast_dataset = fast_dataset.shuffle(TOTAL_SAMPLES, reshuffle_each_iteration=True)
    
    real_ds = fast_dataset.filter(lambda f, l: l[0] == 0).map(lambda f, l: f)
    fake_ds = fast_dataset.filter(lambda f, l: l[0] == 1).map(lambda f, l: f)
    
    adversarial_ds = tf.data.Dataset.zip(
        (real_ds.batch(BATCH_SIZE, drop_remainder=True), 
         fake_ds.batch(BATCH_SIZE, drop_remainder=True))
    ).prefetch(tf.data.AUTOTUNE)
    
    classifier_train_ds = fast_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    # --- 모델 생성 및 컴파일 (기존 동일) ---
    teacher = create_gendet_component(input_dim=FEATURE_DIM, model_dim=196, num_heads=8, ff_dim=784, name="teacher_v4")
    student = create_gendet_component(input_dim=FEATURE_DIM, model_dim=196, num_heads=8, ff_dim=784, name="student_v4")
    augmenter = create_augmenter(input_dim=FEATURE_DIM, name="augmenter_v4")
    #classifier = create_classifier(name="classifier_v3")
    classifier = create_classifier(input_dim=3, name="classifier_v3")

    initial_learning_rate = 5e-5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)

    optimizers_config = {
        't_optimizer': optimizers.Adam(learning_rate=lr_schedule),
        's_optimizer': optimizers.Adam(learning_rate=lr_schedule),
        'a_optimizer': optimizers.Adam(learning_rate=1e-8), # 약간 높임 (긴장감 조성)
        'c_optimizer': optimizers.Adam(learning_rate=1e-3),
    }

    # --- 학습 실행 ---
    # ★★★ Margin을 1.0으로 낮춰서 학생이 포기하지 않게 함
    gendet_trainer = GenDet(teacher, student, augmenter, classifier, margin=0.5) 
    gendet_trainer.compile(**optimizers_config)
    
    print("\n--- [Step 0] Pre-training Teacher ---")
    gendet_trainer.pretrain_teacher(classifier_train_ds, epochs=5) 

    print("\n--- [Step 1-3] Adversarial Training (Long Run) ---")
    # Epoch를 30으로 대폭 늘림 (Fast Mode라 금방 끝남)
    gendet_trainer.fit(adversarial_ds, epochs=4)
    
    print("\n--- [Step 4] Classifier Training ---")
    # Classifier도 충분히 학습
    gendet_trainer.train_classifier(classifier_train_ds, epochs=10)
    
    print("\n--- All Training Finished ---")

    # --- 모델 저장 ---
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"gendet_saved_models_{current_time}"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    teacher.save(os.path.join(save_dir, "teacher.keras"))
    student.save(os.path.join(save_dir, "student.keras"))
    classifier.save(os.path.join(save_dir, "classifier.keras"))
    augmenter.save(os.path.join(save_dir, "augmenter.keras"))

    print(f" Models saved to {save_dir}")