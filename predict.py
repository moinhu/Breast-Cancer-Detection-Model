from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model once, global scope
MODEL_PATH = "breast_cancer_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

def predict_image(img_path):
    try:
        # Load & preprocess
        img = image.load_img(img_path, target_size=(128,128))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)[0][0]

        if pred < 0.5:
            result = "Prediction: Benign (0)"
        else:
            result = "Prediction: Malignant (1)"

        print(result)  # Terminal ke liye
        return result

    except Exception as e:
        print("Prediction failed:", e)
        return "Prediction failed"
