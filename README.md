
#  Voice-Based Speaker Recognition +  Text-to-Speech System

This project demonstrates how to use **Text-to-Speech (TTS)** and **Voice Recognition** using classical machine learning (SVM). It generates voice samples using `pyttsx3`, extracts features using `librosa`, and builds a speaker classifier using **Support Vector Machines**. It also contains a snippet for an **Arduino-based smart traffic system** using ultrasonic sensors.

---

##  Features

-  Convert text into speech (.wav files)
-  Use two different TTS voices (male/female)
-  Extract MFCC audio features using  `librosa`
-  Train a classifier to identify which speaker is talking
-  Test unknown voice to predict the speaker
-  Bonus: Arduino traffic light controller using ultrasonic sensors

---

##  Project Structure

```
voice-recognition/
│
├── voices/                 # Folder storing generated speech WAV files
│   ├── speaker1.wav
│   ├── speaker2.wav
│   └── test_speaker1.wav
│
├── main.py                 # Main script for generating voice, training, and testing
├── README.md               # You're reading it!
```

---

##  Getting Started

### 1. 🔧 Install Requirements

```bash
pip install pyttsx3 librosa soundfile scikit-learn numpy
```

> You also need to have a TTS voice engine (like SAPI5 on Windows).

---

### 2.  Generate Voices

In the code:
```python
generate_speech("Hello, this is speaker one...", "voices/speaker1.wav", voice_1)
generate_speech("Hello, this is speaker two...", "voices/speaker2.wav", voice_2)
```

This creates `.wav` files using two different voice IDs.

---

### 3.  Extract Features & Train Classifier

We extract **MFCCs** (Mel-frequency cepstral coefficients) from audio, and train a **Support Vector Machine** to distinguish speakers.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
```

---

### 4.  Predict Speaker from New Voice

```python
generate_speech("This is a test phrase...", "voices/test_speaker1.wav", voice_1)
test_mfccs = extract_features("voices/test_speaker1.wav")
predictions = clf.predict(test_mfccs)
```

> The model outputs the most likely speaker (Speaker 1 or 2).

---

##  Speak Any Text in Real-Time

```python
speak("How are you buddy, what's going on?", voice_1)
```

This speaks the given sentence in real time.

---
##  License

This project is open source and free to use under the **VIT license**.

---

##  Acknowledgements

- [`pyttsx3`](https://pypi.org/project/pyttsx3/)
- [`librosa`](https://librosa.org/)
- [`scikit-learn`](https://scikit-learn.org/)
- [Arduino Docs](https://www.arduino.cc/reference/en/)

---

## 🙌 Author

Made with ❤️ by **VeerDhawal**  
> Feel free to fork, contribute, or suggest improvements!
