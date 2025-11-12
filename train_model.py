from data_preprocessing import train_generator, val_generator, test_generator
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =============================
# Build Transfer Learning Model
# =============================
def build_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128,128,3))
    base_model.trainable = False  # freeze base layers initially

    x = Flatten()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# =============================
# Callbacks
# =============================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss")
]

# =============================
# Class Weights
# =============================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# =============================
# Stage 1: Train top layers
# =============================
model = build_model()
print("Training top layers (frozen ResNet)...")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# =============================
# Stage 2: Fine-tune deeper layers
# =============================
print("Fine-tuning last 10 layers of ResNet...")

for layer in model.layers[-10:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="binary_crossentropy",
              metrics=["accuracy"])

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save final fine-tuned model
model.save("breast_cancer_model.h5")
print("✅ Fine-tuned model saved as breast_cancer_model.h5")

# =============================
# Evaluation on Test Data
# =============================
print("\nEvaluating on test set...")
y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype(int).ravel()
y_true = test_generator.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=["Benign", "Malignant"]))

cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.close()

# Training Curves
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"] + history_finetune.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"] + history_finetune.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.savefig("training_curve.png")
plt.close()

print("✅ Evaluation plots saved (roc_curve.png, training_curve.png)")
