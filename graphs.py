import numpy as np
import matplotlib.pyplot as plt

emotions = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

precision = [0.46,0.78,0.44,0.78,0.49,0.44,0.72]
recall    = [0.51,0.19,0.32,0.80,0.60,0.43,0.73]

x = np.arange(len(emotions))
width = 0.35

plt.bar(x - width/2, precision, width, label="Precision")
plt.bar(x + width/2, recall, width, label="Recall")

plt.xticks(x, emotions)
plt.xlabel("Emotion Classes")
plt.ylabel("Score")
plt.title("Precision vs Recall (Ordered Emotions)")
plt.legend()

plt.show()