import librosa
import numpy as np
import os

dataset_path = "ravdess"
emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

MAX_LEN = 130   # fixed time frames

X, y = [], []

for folder in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, folder)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.lower().endswith(".wav"):
            continue

        parts = file.split("-")
        if len(parts) < 3:
            continue

        emotion_code = parts[2]
        if emotion_code not in emotions:
            continue

        try:
            audio, sr = librosa.load(
                os.path.join(actor_path, file),
                duration=3,
                offset=0.5
            )

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc = mfcc.T

            if mfcc.shape[0] < MAX_LEN:
                pad_width = MAX_LEN - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
            else:
                mfcc = mfcc[:MAX_LEN]

            X.append(mfcc)
            y.append(emotions[emotion_code])

        except:
            continue

X = np.array(X)
y = np.array(y)

np.save("X_speech.npy", X)
np.save("y_speech.npy", y)

print("Samples:", X.shape[0])
