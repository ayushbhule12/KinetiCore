import cv2
import mediapipe as mp
import urllib.request
import os
import math
import threading
import pyttsx3

# ════════════════════════════════════════════════════════
#  TTS ENGINE (runs in background thread so it doesn't
#  freeze the camera while speaking)
# ════════════════════════════════════════════════════════

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)   # speaking speed
tts_lock   = threading.Lock()

def speak(text):
    """Speak text in a background thread — never blocks the camera loop."""
    def _speak():
        with tts_lock:
            tts_engine.say(text)
            tts_engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# ════════════════════════════════════════════════════════
#  WORD SUGGESTIONS
#  Maps a sign → list of word suggestions shown on screen.
#  Press 1/2/3 to pick one and append it to detected text.
# ════════════════════════════════════════════════════════

SIGN_SUGGESTIONS = {
    "Open Hand"     : ["hello", "hi", "wave"],
    "Thumbs Up"     : ["yes", "good", "ok"],
    "V / Peace"     : ["peace", "victory", "two"],
    "Rock On"       : ["rock", "love", "cool"],
    "Fist"          : ["stop", "no", "power"],
    "Point"         : ["you", "that", "there"],
    "I / Pinky"     : ["I", "me", "eye"],
    "L"             : ["left", "loser", "L"],
    "B"             : ["bye", "B", "back"],
    "W / 3"         : ["wow", "win", "three"],
    "Middle Finger" : ["fuck you", "f*** you", "🖕"],   # ← ADDED
}

# ════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ════════════════════════════════════════════════════════

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (~25MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

latest_result = None

def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    result_callback=on_result
)
detector = vision.HandLandmarker.create_from_options(options)

# ════════════════════════════════════════════════════════
#  DRAW LANDMARKS
# ════════════════════════════════════════════════════════

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 180, 90), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 6, (0, 255, 160), -1)
        cv2.circle(frame, (x, y), 6, (0, 60, 30),   1)

# ════════════════════════════════════════════════════════
#  DISTANCE HELPER
# ════════════════════════════════════════════════════════

def dist(lm, a, b):
    """Normalized distance between landmark a and b."""
    return math.sqrt((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2)

# ════════════════════════════════════════════════════════
#  FINGER STATES
# ════════════════════════════════════════════════════════

def get_finger_states(lm):
    fingers = {}
    fingers['thumb']  = lm[4].x  < lm[3].x
    fingers['index']  = lm[8].y  < lm[6].y
    fingers['middle'] = lm[12].y < lm[10].y
    fingers['ring']   = lm[16].y < lm[14].y
    fingers['pinky']  = lm[20].y < lm[18].y
    return fingers

# ════════════════════════════════════════════════════════
#  FULL ASL CLASSIFIER  (A–Y, excluding J and Z)
# ════════════════════════════════════════════════════════

def classify_sign(lm, fs):
    t = fs['thumb']
    i = fs['index']
    m = fs['middle']
    r = fs['ring']
    p = fs['pinky']

    # ── Pre-compute useful distances ──────────────────
    ti  = dist(lm, 4, 8)   # thumb tip  ↔ index tip
    tm  = dist(lm, 4, 12)  # thumb tip  ↔ middle tip
    tr  = dist(lm, 4, 16)  # thumb tip  ↔ ring tip
    tp  = dist(lm, 4, 20)  # thumb tip  ↔ pinky tip
    im  = dist(lm, 8, 12)  # index tip  ↔ middle tip

    # ── Middle finger gesture ─────────────────────────  ← ADDED
    # Only middle finger up, all others (index, ring, pinky, thumb) down
    if not i and m and not r and not p and not t:   return "Middle Finger"

    # ── Shortcut gestures (whole words) ───────────────
    if i and m and r and p and t:                   return "Open Hand"
    if not i and not m and not r and p and t \
       and tp > 0.40:                               return "Rock On"
    if not i and not m and not r and not p \
       and t and lm[4].y < lm[3].y:                return "Thumbs Up"

    # ── ASL Letters ───────────────────────────────────

    # A: fist, thumb rests on side
    if not i and not m and not r and not p \
       and t and ti > 0.15:                         return "A"

    # B: four fingers straight up, thumb tucked
    if i and m and r and p and not t \
       and im < 0.20:                               return "B"

    # C: all fingers curve into a C
    if not i and not m and not r and not p \
       and not t and ti < 0.40:                     return "C"

    # D: index up, others curl to thumb
    if i and not m and not r and not p \
       and tm < 0.20:                               return "D"

    # E: all fingers curl down tightly
    if not i and not m and not r and not p \
       and not t and ti < 0.18:                     return "E"

    # F: index+thumb touch, other 3 up
    if not i and m and r and p \
       and ti < 0.12:                               return "F"

    # G: index points sideways, thumb parallel
    if i and not m and not r and not p \
       and t and ti > 0.35:                         return "G"

    # H: index+middle point sideways together
    if i and m and not r and not p \
       and not t and im < 0.18:                     return "H"

    # I: pinky only up
    if not i and not m and not r and p \
       and not t:                                   return "I"

    # K: index+middle up, thumb between
    if i and m and not r and not p \
       and t and im < 0.28:                         return "K"

    # L: index up + thumb out
    if i and not m and not r and not p \
       and t:                                       return "L"

    # M: three fingers folded over thumb
    if not i and not m and not r and not p \
       and not t and ti < 0.22 and tm < 0.22:       return "M"

    # N: two fingers folded over thumb
    if not i and not m and not r and p \
       and not t and ti < 0.22:                     return "N"

    # O: all fingers curl into O ring
    if not i and not m and not r and not p \
       and not t and ti < 0.20 and tp < 0.30:       return "O"

    # P: index pointing downward
    if i and m and not r and not p \
       and t and lm[8].y > lm[0].y:                return "P"

    # Q: index+thumb point down
    if i and not m and not r and not p \
       and t and lm[8].y > lm[5].y:                return "Q"

    # R: index+middle crossed tightly
    if i and m and not r and not p \
       and not t and im < 0.12:                     return "R"

    # S: fist with thumb over fingers
    if not i and not m and not r and not p \
       and t and ti < 0.20:                         return "S"

    # T: thumb between index and middle
    if not i and not m and not r and not p \
       and t and ti < 0.18 and tm < 0.22:           return "T"

    # U: index+middle up, close together
    if i and m and not r and not p \
       and not t and im < 0.18:                     return "U"

    # V: index+middle up, spread apart
    if i and m and not r and not p \
       and not t and im > 0.22:                     return "V / Peace"

    # W: index+middle+ring all up
    if i and m and r and not p and not t:           return "W / 3"

    # X: index finger hooked
    if not i and not m and not r and not p \
       and not t and ti < 0.28:                     return "X"

    # Y: thumb + pinky out
    if not i and not m and not r and p \
       and t and tp > 0.40:                         return "Y"

    # Fallback gestures
    if i and not m and not r and not p \
       and not t:                                   return "Point"
    if not i and not m and not r and not p \
       and not t:                                   return "Fist"

    return "Unknown"

# ════════════════════════════════════════════════════════
#  HOLD-TO-CONFIRM STATE
# ════════════════════════════════════════════════════════

detected_text       = []
last_sign           = ""
hold_count          = 0
HOLD_NEEDED         = 20
last_spoken         = ""
current_suggestions = []

# ════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════

cap       = cv2.VideoCapture(0)
timestamp = 0

print("Running... Controls: Q=quit  SPACE=space  C=clear  S=speak  1/2/3=pick suggestion")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam not found.")
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp += 1
    detector.detect_async(mp_image, timestamp)

    sign = "No hand"

    if latest_result and latest_result.hand_landmarks:
        lm   = latest_result.hand_landmarks[0]
        draw_landmarks(frame, lm)

        fs   = get_finger_states(lm)
        sign = classify_sign(lm, fs)

        # ── Hold-to-confirm ───────────────────────────
        if sign == last_sign and sign != "Unknown":
            hold_count += 1
        else:
            hold_count = 0
            last_sign  = sign
            current_suggestions = SIGN_SUGGESTIONS.get(sign, [])

        if hold_count == HOLD_NEEDED:
            detected_text.append(sign)
            speak(sign)

    # ════════════════════════════════════════════════
    #  DRAW HUD
    # ════════════════════════════════════════════════

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 70), (10, 12, 18), -1)

    # Sign label — red for middle finger, green for everything else
    if sign == "Middle Finger":
        color = (60, 60, 255)   # red
    elif sign not in ("No hand", "Unknown"):
        color = (0, 255, 120)   # green
    else:
        color = (80, 80, 100)   # grey

    cv2.putText(frame, sign, (14, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

    # Hold progress bar
    cv2.rectangle(frame, (10, 58), (w - 10, 65), (30, 30, 45), -1)
    if hold_count > 0:
        fill = int((hold_count / HOLD_NEEDED) * (w - 20))
        bar_color = (60, 60, 255) if sign == "Middle Finger" else (0, 255, 120)
        cv2.rectangle(frame, (10, 58), (10 + fill, 65), bar_color, -1)

    # Suggestions panel
    if current_suggestions:
        sx = w - 180
        cv2.rectangle(frame, (sx - 8, 75),
                      (w - 8, 75 + len(current_suggestions) * 36 + 10),
                      (18, 20, 30), -1)
        cv2.putText(frame, "Suggestions:", (sx, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 160), 1)
        for idx, word in enumerate(current_suggestions):
            y = 115 + idx * 34
            cv2.rectangle(frame, (sx - 4, y - 20), (w - 12, y + 8), (25, 30, 45), -1)
            cv2.putText(frame, f"{idx+1}. {word}", (sx, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 140), 2)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 60), (w, h), (10, 12, 18), -1)
    cv2.putText(frame, "Q=quit  SPC=space  C=clear  S=speak  1/2/3=word",
                (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 100), 1)

    text_display = " ".join(detected_text[-12:])
    cv2.putText(frame, text_display if text_display else "...",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    cv2.imshow("Hand Sign Detector", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if   key == ord('q'):
        break
    elif key == ord(' '):
        detected_text.append(' ')
    elif key == ord('c'):
        detected_text = []
    elif key == ord('s'):
        sentence = " ".join(detected_text)
        if sentence.strip():
            speak(sentence)
    elif key == ord('1') and len(current_suggestions) >= 1:
        word = current_suggestions[0]
        detected_text.append(word)
        speak(word)
    elif key == ord('2') and len(current_suggestions) >= 2:
        word = current_suggestions[1]
        detected_text.append(word)
        speak(word)
    elif key == ord('3') and len(current_suggestions) >= 3:
        word = current_suggestions[2]
        detected_text.append(word)
        speak(word)

cap.release()
cv2.destroyAllWindows()