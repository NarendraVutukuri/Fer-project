import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from flask import Flask, render_template
import multiprocessing

from speech.speech_emotion_realtime import detect_speech_emotion
from face.face_emotion_realtime import detect_face_emotion
from fusion.adaptive_fusion import fuse
from response_bot import generate_response

app = Flask(__name__)


# ---------- ROUTES ----------

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/speech')
def speech():
    return render_template("speech.html")

@app.route('/face')
def face():
    return render_template("face.html")

@app.route('/multimodal')
def multimodal():
    return render_template("multimodal.html")


# ---------- PROCESS FUNCTIONS ----------

def run_speech(q):
    try:
        result = detect_speech_emotion()
        q.put(result)
    except:
        q.put({"neutral": 1.0})


def run_face(q):
    try:
        result = detect_face_emotion()
        q.put(result)
    except:
        q.put({"neutral": 1.0})


# ---------- DETECTION ROUTES ----------

@app.route('/detect_speech')
def detect_speech():
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_speech, args=(q,))
    p.start()
    p.join()

    probs = q.get()
    emotion = max(probs, key=probs.get)
    response = generate_response(emotion)

    return f"<h2>Speech Emotion: {emotion}</h2><p>{response}</p>"


@app.route('/detect_face')
def detect_face():
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_face, args=(q,))
    p.start()
    p.join()

    probs = q.get()
    emotion = max(probs, key=probs.get)
    response = generate_response(emotion)

    return f"<h2>Face Emotion: {emotion}</h2><p>{response}</p>"


@app.route('/detect_multimodal')
def detect_multimodal():
    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=run_speech, args=(q1,))
    p2 = multiprocessing.Process(target=run_face, args=(q2,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    speech_probs = q1.get()
    face_probs = q2.get()

    final_probs = fuse(face_probs, speech_probs)
    final_emotion = max(final_probs, key=final_probs.get)

    response = generate_response(final_emotion)

    return f"<h2>Final Emotion: {final_emotion}</h2><p>{response}</p>"


# ---------- RUN ----------

if __name__ == '__main__':
    app.run(debug=False)