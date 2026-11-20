import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# LOAD MODEL
model = load_model("models/face_emotion_model.h5")

# DATA
test_dir = "fer2013/test"

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

# PREDICTION
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes

labels = list(test_data.class_indices.keys())

# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)
plt.title("Face Emotion Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# CLASSIFICATION REPORT
print(classification_report(y_true, y_pred_classes, target_names=labels))
