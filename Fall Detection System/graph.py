import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# 🔥 LOAD HISTORIES
history_A = pickle.load(open("history_A.pkl","rb"))
history_B = pickle.load(open("history_B.pkl","rb"))
history_C = pickle.load(open("history_C.pkl","rb"))

# 🔥 LOAD MODELS
model_A = tf.keras.models.load_model("model_A.h5")
model_B = tf.keras.models.load_model("model_B.h5")
model_C = tf.keras.models.load_model("model_C.h5")

# 🔥 VALIDATION DATA
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

val_generator = datagen.flow_from_directory(
    r"D:\frames\val",
    target_size=(128,128),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

y_true = val_generator.classes

# =========================
# 📊 1. DATASET OVERVIEW
# =========================
labels, counts = np.unique(y_true, return_counts=True)

plt.figure()
plt.bar(['Fall','Non-Fall'], counts)
plt.title("Dataset Overview")
plt.show()

# =========================
# 📊 2. TRAINING HISTORY
# =========================
def plot_history(history, title):
    plt.figure()
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.title(f"{title} Accuracy")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f"{title} Loss")
    plt.legend()
    plt.show()

plot_history(history_A, "Model A (MobileNetV2)")
plot_history(history_B, "Model B (Custom CNN)")
plot_history(history_C, "Model C (EfficientNetB0)")

# =========================
# 📊 3. ROC CURVES
# =========================
def compute_roc(model, name):
    y_pred = model.predict(val_generator).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

fpr_A, tpr_A, auc_A = compute_roc(model_A, "A")
fpr_B, tpr_B, auc_B = compute_roc(model_B, "B")
fpr_C, tpr_C, auc_C = compute_roc(model_C, "C")

plt.figure()
plt.plot(fpr_A, tpr_A, label=f"Model A (AUC={auc_A:.2f})")
plt.plot(fpr_B, tpr_B, label=f"Model B (AUC={auc_B:.2f})")
plt.plot(fpr_C, tpr_C, label=f"Model C (AUC={auc_C:.2f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# =========================
# 📊 4. CONFUSION MATRIX
# =========================
def plot_cm(model, name):
    y_pred = (model.predict(val_generator) > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    # counts
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"{name} Confusion Matrix (Counts)")
    plt.show()

    # normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    sns.heatmap(cm_norm, annot=True, fmt='.2f')
    plt.title(f"{name} Confusion Matrix (Normalized)")
    plt.show()

plot_cm(model_A, "Model A")
plot_cm(model_B, "Model B")

# =========================
# 📊 5. PERFORMANCE BAR
# =========================
acc_A = max(history_A['val_accuracy'])
acc_B = max(history_B['val_accuracy'])
acc_C = max(history_C['val_accuracy'])

plt.figure()
plt.bar(['Model A','Model B','Model C'], [acc_A, acc_B, acc_C])
plt.title("Performance Comparison (Validation Accuracy)")
plt.show()

print("✅ All graphs generated successfully")