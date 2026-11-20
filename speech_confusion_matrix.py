import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# LOAD DATA
X = np.load("X_speech.npy")
y = np.load("y_speech.npy", allow_pickle=True)

# LOAD MODEL
model = load_model("models/speech_emotion_model.h5")

# EMOTION LABELS (MUST MATCH TRAINING)
emotion_labels = [
    "NEUTRAL",
    "CALM",
    "HAPPY",
    "SAD",
    "ANGRY",
    "FEARFUL",
    "DISGUST",
    "SURPRISED"
]

# CONVERT STRING LABELS → NUMBERS
label_to_index = {label: i for i, label in enumerate(emotion_labels)}
y_numeric = np.array([label_to_index[label] for label in y])

# SPLIT SAME AS TRAINING
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
)

# PREDICTIONS
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_classes)

# PLOT
plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=emotion_labels,
    yticklabels=emotion_labels
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Speech Emotion Confusion Matrix")
plt.tight_layout()
plt.show()

# CLASSIFICATION REPORT
print(classification_report(y_test, y_pred_classes, target_names=emotion_labels))
