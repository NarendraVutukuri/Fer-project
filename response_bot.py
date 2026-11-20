import pyttsx3
import random

responses = {
    "joy": [
        "That's wonderful to hear!",
        "I'm really happy for you!",
        "You sound joyful—keep smiling!",
        "That's a great mood to be in!",
        "Awesome! What's making you so happy?"
    ],

    "sad": [
        "I'm here for you. Things will get better.",
        "It's okay to feel sad sometimes.",
        "Do you want to talk about it?",
        "Take your time—I'm listening.",
        "You’re not alone in this."
    ],

    "angry": [
        "Take a deep breath. You’ve got this.",
        "It might help to pause for a moment.",
        "Try to relax and let it go slowly.",
        "I understand that you're upset.",
        "Want to talk about what's bothering you?"
    ],

    "fear": [
        "You're safe. Take it one step at a time.",
        "It’s okay to feel scared sometimes.",
        "I’m here with you.",
        "Try to stay calm—you’ll get through this.",
        "Would you like to share what’s worrying you?"
    ],

    "surprise": [
        "That sounds unexpected!",
        "Wow, that’s surprising!",
        "Didn't see that coming, huh?",
        "That must have caught you off guard!",
        "Tell me more about it!"
    ],

    "disgust": [
        "That doesn't sound pleasant.",
        "I understand why you'd feel that way.",
        "That must be uncomfortable.",
        "Yeah, that can be quite unpleasant.",
        "Want to talk about what caused that feeling?"
    ],

    "neutral": [
        "Tell me more about how you're feeling.",
        "I'm listening.",
        "Go ahead, I'm here.",
        "Would you like to share something?",
        "How has your day been so far?"
    ]
}


def generate_response(emotion):
    text = random.choice(responses.get(emotion, responses["neutral"]))

    try:
        # ✅ create fresh engine every time
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()   # ✅ important cleanup

    except Exception as e:
        print("TTS Error:", e)

    return text