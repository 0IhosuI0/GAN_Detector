import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
import numpy as np
import base64
import io
import os
from PIL import Image, ImageOps
from transformers import TFCLIPVisionModel, CLIPProcessor
import cv2

app = FastAPI()

class ImageRequest(BaseModel):
    filename: str
    image_base64: str

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


print("Loading AI Model...")

RESNET_MODEL_PATH = "best_model_EPOCH50.h5"
GENDET_MODEL_PATH = "gendet/"

TEACHER_PATH = os.path.join(GENDET_MODEL_PATH, "teacher.keras")
STUDENT_PATH = os.path.join(GENDET_MODEL_PATH, "student.keras")
CLASSIFIER_PATH = os.path.join(GENDET_MODEL_PATH, "classifier.keras")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
GENDET_IMG_SIZE = 224

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(pil_image):
    try:
        img_np = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))

        if len(faces > 0):
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_crop_bgr = img_bgr[y:y+h, x:x+w]

            face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            return True, Image.fromarray(face_crop_rgb)
        else:
            return False, None
    except Exception as e:
        print(f"[Error] Face detection error: {e}")
        return False, None

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"GPU Error: {e}")
    
    model = tf.keras.models.load_model(RESNET_MODEL_PATH)
    print("ResNet Model loaded successfully")

    custom_objects = {"TransformerBlock": TransformerBlock}
    teacher_model = keras.models.load_model(TEACHER_PATH, custom_objects=custom_objects, compile=False)
    student_model = keras.models.load_model(STUDENT_PATH, custom_objects=custom_objects, compile=False)
    classifier_model = keras.models.load_model(CLASSIFIER_PATH, custom_objects=custom_objects, compile=False)
    print("GenDet Model Loaded")
    
    print(f"Loading CLIP Backbone...")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_vision_model = TFCLIPVisionModel.from_pretrained(CLIP_MODEL_NAME)
    print("CLIP loaded.")


except Exception as e:
    print(f"Error Loading Model : {e}")
    model = None

def preprocess_for_resnet_model(image):
    try:
        
        #image = image.resize((256, 256))
        image = ImageOps.pad(image, (256, 256), color='black')

    except Exception as e:
        return {"error": f"Invalid image data: {str(e)}"}
    
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    return img_array

def preprocess_for_gendet_model(image_bytes):
    img_bytes_tf = tf.convert_to_tensor(image_bytes, dtype=tf.string)
    img = tf.image.decode_image(img_bytes_tf, channels=3, expand_animations=False)
    img = tf.image.resize(img, [GENDET_IMG_SIZE, GENDET_IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    
    img_np = img.numpy()
    img_uint8 = (img_np * 255).astype(np.uint8)
    
    inputs = clip_processor(images=img_uint8, return_tensors="tf", padding=True)
    return inputs["pixel_values"]

@app.post("/predict")
def predict_image(request: ImageRequest):
    if model is None:
        return {"error": "Model is not loaded."}
    
    
    image_bytes = base64.b64decode(request.image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    has_face, crop_image = detect_faces(image)

    if has_face:
        resnet_img_array = preprocess_for_resnet_model(crop_image)

        resnet_prediction = model.predict(resnet_img_array)
        resnet_score = float(resnet_prediction[0][0])
        
        try:
            pixel_values = preprocess_for_gendet_model(image_bytes)
            #print(f"[CLIP Input] Pixel Values Shape: {pixel_values.shape}")
            # CLIP Feature 추출
            clip_out = clip_vision_model(pixel_values=pixel_values)
            features = clip_out.pooler_output
            #print(f"[CLIP Output] Features Shape: {features.shape}")

            # GenDet Flow
            z_t = teacher_model(features, training=False)
            z_s = student_model(features, training=False)
            abs_diff = tf.abs(z_t - z_s)
            
            # Classifier 결과 (Fake일 확률)
            gendet_score = float(classifier_model(abs_diff, training=False).numpy().reshape(-1)[0])
        except Exception as e:
            return {"error": f"GenDet prediction failed: {e}"}
        
        
        final_score = 0.7 * resnet_score + 0.3 * gendet_score

        if final_score > 0.6:
            result_label = 1 # Fake
            confidence = final_score
        else:
            result_label = 0 # Real
            confidence = final_score

        print(f"filename: {request.filename}, prediction: {result_label}, confidence: {round(confidence, 4)}, gendet_score: {gendet_score}, resnet_score: {resnet_score}")
    else:
        try:
            pixel_values = preprocess_for_gendet_model(image_bytes)
            #print(f"[CLIP Input] Pixel Values Shape: {pixel_values.shape}")
            # CLIP Feature 추출
            clip_out = clip_vision_model(pixel_values=pixel_values)
            features = clip_out.pooler_output
            #print(f"[CLIP Output] Features Shape: {features.shape}")

            # GenDet Flow
            z_t = teacher_model(features, training=False)
            z_s = student_model(features, training=False)
            abs_diff = tf.abs(z_t - z_s)
            
            # Classifier 결과 (Fake일 확률)
            final_score = float(classifier_model(abs_diff, training=False).numpy().reshape(-1)[0])
        except Exception as e:
            return {"error": f"GenDet prediction failed: {e}"}
        
        if final_score > 0.6:
            result_label = 1 # Fake
            confidence = final_score
        else:
            result_label = 0 # Real
            confidence = final_score
        
        print(f"filename: {request.filename}, prediction: {result_label}, confidence: {round(confidence, 4)}")
        

    

    # Debug
    #print(f"filename: {request.filename}, prediction: {result_label}, confidence: {round(confidence, 4)}, gendet_score: {gendet_score}, resnet_score: {resnet_score}")

    return {
        "filename": request.filename,
        "prediction": result_label,
        "confidence": round(confidence, 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=35480)