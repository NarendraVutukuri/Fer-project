import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load extracted MFCC features
X = np.load("X_speech.npy")
y = np.load("y_speech.npy")

print("Feature shape:", X.shape)
print("Labels shape:", y.shape)

# Encode emotion labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)

# Add channel dimension for CNN
X = X[..., np.newaxis]   # (samples, 130, 40, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(130, 40, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model and labels
model.save("models/speech_emotion_model.h5")
np.save("speech_labels.npy", encoder.classes_)

print("✅ Speech emotion model trained and saved successfully")
