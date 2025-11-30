import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image, ImageOps

app = FastAPI()

class ImageRequest(BaseModel):
    filename: str
    image_base64: str

print("Loading AI Model...")

MODEL_PATH = "best_model_EPOCH100.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")

except Exception as e:
    print(f"Error Loading Model : {e}")
    model = None

@app.post("/predict")
def predict_image(request: ImageRequest):
    if model is None:
        return {"error": "Model is not loaded."}
    
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        #image = image.resize((256, 256))
        image = ImageOps.pad(image, (256, 256), color='black')

    except Exception as e:
        return {"error": f"Invalid image data: {str(e)}"}
    
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    score = float(prediction[0][0])

    if score > 0.5:
        result_label = 1 # Fake
        confidence = score
    else:
        result_label = 0 # Real
        confidence = score

    return {
        "filename": request.filename,
        "prediction": result_label,
        "confidence": round(confidence, 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=35480)