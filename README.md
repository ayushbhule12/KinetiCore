# 🤚 Kineticore — ASL to Text

A real-time hand gesture recognition system that uses your laptop webcam to detect hand signs and convert them to text, with text-to-speech output and word suggestions.

Built with Python, MediaPipe, OpenCV, and pyttsx3.

---

## 📸 What It Does

- Detects your hand in real-time via webcam
- Recognizes **ASL letters A–Y** (J and Z excluded — they need motion)
- Recognizes **common gestures** like Thumbs Up, Peace, Rock On, Middle Finger
- **Hold-to-confirm**: hold a sign steady for ~0.6s to commit it to text
- **Word suggestions**: press 1/2/3 to insert whole words instantly
- **Text-to-speech**: speaks every committed sign out loud
- Green progress bar fills as you hold a sign — turns red for Middle Finger 🖕

---

## 🗂️ Project Structure

```
hand-sign/
├── main.py               ← entire application (single file)
├── requirements.txt      ← Python dependencies
└── hand_landmarker.task  ← auto-downloaded on first run (~25MB)
```

---

## ⚙️ Requirements

- **Python 3.10** (required — MediaPipe does not support 3.12/3.13 on Windows)
- **Conda** (recommended for managing Python versions)
- A working webcam

---

## 🚀 Setup — Step by Step

### 1. Create a Python 3.10 Conda environment

```bash
conda create -n handsign python=3.10
conda activate handsign
```

> You must use the `handsign` environment every time you work on this project.
> If your terminal shows `(base)` instead of `(handsign)`, run `conda activate handsign` first.

### 2. Install dependencies

```bash
pip install opencv-python mediapipe==0.10.14 numpy pyttsx3
```

### 3. Run the app

```bash
python main.py
```

On first run, the app will automatically download the MediaPipe hand landmark model (~25MB). This only happens once.

---

## 🖐 Supported Signs

### ASL Letters

| Letter | Hand Shape |
|--------|-----------|
| A | Fist, thumb resting on side |
| B | Four fingers up, thumb tucked in |
| C | Curved hand like holding a ball |
| D | Index up, others curl to meet thumb |
| E | All fingers curled tightly down |
| F | Index+thumb touch, other 3 up |
| G | Index+thumb pointing sideways |
| H | Index+middle pointing sideways together |
| I | Pinky only up |
| K | Index+middle up, thumb between them |
| L | Index up + thumb out (L shape) |
| M | Three fingers folded over thumb |
| N | Two fingers folded over thumb |
| O | All fingers curl into an O ring |
| P | Index pointing downward |
| Q | Index+thumb pointing down |
| R | Index+middle crossed tightly |
| S | Fist, thumb over fingers |
| T | Thumb tucked between index+middle |
| U | Index+middle up, close together |
| V | Index+middle up, spread apart ✌️ |
| W | Index+middle+ring up and spread |
| X | Index finger hooked |
| Y | Thumb+pinky out 🤙 |

> J and Z are excluded — they require hand motion, not a static pose.

### Gesture Shortcuts (Whole Words)

| Gesture | Suggestions |
|---------|------------|
| 🖐 Open Hand | hello, hi, wave |
| 👍 Thumbs Up | yes, good, ok |
| ✌️ V / Peace | peace, victory, two |
| 🤘 Rock On | rock, love, cool |
| ✊ Fist | stop, no, power |
| ☝️ Point | you, that, there |
| 🖕 Middle Finger | fuck you, f*** you, 🖕 |
| 🤙 I / Pinky | I, me, eye |
| 👆 L | left, loser, L |
| 🅱️ B | bye, B, back |
| 3️⃣ W / 3 | wow, win, three |

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `Space` | Add a space to the text |
| `C` | Clear all detected text |
| `S` | Speak the full sentence out loud |
| `1` | Insert first word suggestion |
| `2` | Insert second word suggestion |
| `3` | Insert third word suggestion |
| `Q` | Quit the app |

---

## 🧠 How It Works — The Pipeline

```
Webcam frame
    ↓
MediaPipe Hand Landmarker
    → detects 21 landmarks on your hand (fingertips, knuckles, wrist)
    ↓
Finger State Detection
    → checks if each finger is UP or DOWN
    → thumb: compares X position (moves sideways)
    → others: compare Y position (tip above knuckle = up)
    ↓
Sign Classifier
    → Layer 1: finger state fingerprint (which fingers are up/down)
    → Layer 2: tip-to-tip distances (separates similar signs like U vs V)
    ↓
Hold-to-Confirm
    → sign must be held for 20 frames (~0.6s) before committing
    → prevents accidental triggers from passing gestures
    ↓
Output
    → sign name added to detected text
    → text-to-speech speaks the sign
    → word suggestions shown on screen
```

---

## 🛠 Troubleshooting

| Problem | Fix |
|---------|-----|
| `mediapipe has no attribute 'solutions'` | Wrong mediapipe version. Run `pip install mediapipe==0.10.14` |
| `Could not find mediapipe==0.10.14` | Wrong Python version. Must use Python 3.10 via conda |
| `function 'free' not found` | Python 3.12/3.13 not supported. Create conda env with Python 3.10 |
| Webcam not opening | Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in main.py |
| Signs keep misreading | Improve lighting — face a lamp or window. Keep hand centered in frame |
| `No hand` showing too often | Keep your full hand visible, not cut off at frame edges |
| TTS not speaking | Run `pip install pyttsx3 pywin32` |
| Still in wrong environment | Run `conda activate handsign` then check `where python` |

---

## 📦 Dependencies Explained

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | latest | Webcam access and drawing on frames |
| `mediapipe` | 0.10.14 | Hand landmark detection neural net |
| `numpy` | latest | Array operations for image processing |
| `pyttsx3` | latest | Offline text-to-speech engine |

---

## 🔧 Customization

### Change hold sensitivity
In `main.py`, find:
```python
HOLD_NEEDED = 20   # ~0.6 seconds at 30fps
```
- Increase (e.g. `30`) → harder to accidentally trigger
- Decrease (e.g. `10`) → faster but more accidental commits

### Add your own word suggestions
In `main.py`, find the `SIGN_SUGGESTIONS` dictionary and add your own:
```python
SIGN_SUGGESTIONS = {
    "Thumbs Up" : ["yes", "good", "ok"],   # ← change these words
    "Your Sign" : ["word1", "word2", "word3"],  # ← add new entries
}
```

### Add a new gesture
In `classify_sign()`, add a new rule using the finger state pattern:
```python
# t=thumb, i=index, m=middle, r=ring, p=pinky
# True = finger up, False = finger down
if not i and m and not r and not p and not t:   return "Middle Finger"
```
Then add it to `SIGN_SUGGESTIONS` with its words.

---

## 📝 Issues Encountered During Development

1. **mediapipe version conflict** — `mp.solutions` was removed in mediapipe 0.11+. Fixed by pinning to `0.10.14`.
2. **Python 3.13 incompatibility** — MediaPipe's C bindings (`free` function) fail on Python 3.12/3.13 on Windows. Fixed by creating a Python 3.10 conda environment.
3. **Duplicate function definition** — accidentally had two `get_finger_states` functions in the file from mixing code versions. Fixed by replacing the entire file with a clean version.
4. **`global` keyword inside while loop** — `global hold_count, last_sign` inside a `while` loop causes a Pylance error. Removed it — `global` is only needed inside functions, not loops.

---

## 🔮 Possible Next Steps

- **More ASL letters** — use tip-to-tip distance math to better separate similar signs
- **Backspace gesture** — assign a gesture to delete the last committed sign
- **Save to file** — auto-save detected text to a `.txt` file
- **Web UI** — replace the OpenCV window with a browser interface using Flask
- **ML classifier** — train a real neural net on your own hand data for higher accuracy# KinetiCore
