from tensorflow.keras.models import load_model
from data_preprocessing import test_generator

# Load trained model
model = load_model("breast_cancer_model.h5")

# Evaluate
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")
