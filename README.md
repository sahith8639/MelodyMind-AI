# MelodyMind AI 🎵🧠
### *Compose Infinite Melodies with Artificial Intelligence*

**MelodyMind AI** (Repository: `CodeAlpha_Music_Generation_AI`) is an end-to-end deep learning web application built to train LSTM (Long Short-Term Memory) Recurrent Neural Networks on musical structures and generate original note and chord sequences. 

The application is written in **Python (Flask)** for the backend and utilizes a modern, responsive **Tailwind CSS** white and red themed interface on the frontend, featuring real-time audio visualization powered by **WaveSurfer.js**.

---

## ✨ Features

- 🎹 **MIDI Preprocessing Engine**: Parses MIDI files using the `music21` library to extract notes and multi-pitch chords.
- 🧠 **LSTM Neural Network**: Double-layered LSTM network built using TensorFlow/Keras that learns temporal dependencies in musical motifs.
- ⚡ **Asynchronous Background Training**: Initiates model training from the browser interface without locking the web server thread.
- 📊 **Real-time Metrics Dashboard**: Monitors training accuracy, loss, and progress with live log feeds and Chart.js line charts.
- 🎛️ **Creativity (Temperature) Control**: Modulates composition randomness/creativity via a temperature-based Softmax sampler.
- 🌊 **Waveform Player**: Synthesizes generated MIDI into WAV formats in the backend for immediate in-browser playing and audio rendering with WaveSurfer.js.
- 📥 **Composition Archiver**: Keeps a structured download history of previously generated tracks.
- 🚀 **Production-Ready**: Configured for direct deployment on Render, complete with `Procfile` and `runtime.txt`.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.11, Flask
- **Deep Learning / AI**: TensorFlow, NumPy, Pandas, scikit-learn, music21
- **Audio Synthesis**: pretty_midi, scipy, wave
- **Frontend**: HTML5, Vanilla CSS, Tailwind CSS, Font Awesome, WaveSurfer.js, Chart.js

---

## 📁 Repository Structure

```
MelodyMind-AI/
├── app.py                      # Flask Application Server & Routing
├── train_model.py              # Background Model Trainer
├── generate_music.py           # Music Generation Script
├── preprocess.py               # MIDI data parsing & representation utilities
├── requirements.txt            # Python dependencies
├── Procfile                    # Deployment instructions for Render
├── runtime.txt                 # Target Python runtime version
├── README.md                   # Project documentation
├── .gitignore                  # Git untracked patterns
├── model/                      # Holds model binaries, history, & notes pickle
│   ├── music_model.h5
│   ├── notes.pkl
│   └── training_history.json
├── dataset/                    # Training dataset
│   └── midi_files/             # Initial sample files and user-uploaded files
├── generated/                  # Generated MIDI and synthesized WAV history
└── static/                     # Static frontend files
    ├── css/
    │   └── style.css
    ├── js/
    │   └── script.js
    └── assets/                 # Image assets (hero, logo)
```

---

## 🚀 Local Installation & Setup

### Prerequisites
Ensure you have **Python 3.11.x** and **pip** installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/CodeAlpha_Music_Generation_AI.git
cd CodeAlpha_Music_Generation_AI
```

### 2. Create and Activate a Virtual Environment
**Windows**:
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to view the dashboard!

---

## 🧠 Model & Preprocessing Pipeline

### Data Preprocessing (`preprocess.py`)
- Reads MIDI files and filters events into notes and chords.
- Single notes are mapped to their pitch strings (e.g., `G4`).
- Chords are sorted and mapped to dot-separated pitch strings (e.g., `C4.E4.G4`).
- Elements are mapped to integers and saved to `model/notes.pkl`.
- Input sequence length: `100`. The network uses 100 notes to predict the 101st.

### Network Architecture (`train_model.py`)
```
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_1 (LSTM)                (None, 100, 512)          1,073,152 
dropout_1 (Dropout)          (None, 100, 512)          0         
lstm_2 (LSTM)                (None, 512)               2,099,200 
dense_1 (Dense - ReLU)       (None, 256)               131,328   
dropout_2 (Dropout)          (None, 256)               0         
dense_2 (Dense - Softmax)    (None, vocab_size)        Varies    
=================================================================
```

---

## 📦 Deployment Guide

### Deploying to GitHub
1. Create a public repository on GitHub named `CodeAlpha_Music_Generation_AI`.
2. Initialize and push your project:
   ```bash
   git init
   git add .
   git commit -m "feat: complete end-to-end MelodyMind AI system"
   git branch -M main
   git remote add origin https://github.com/your-username/CodeAlpha_Music_Generation_AI.git
   git push -u origin main
   ```

### Deploying to Render
1. Log in to **Render** (`https://render.com`) and create a new **Web Service**.
2. Connect your GitHub repository `CodeAlpha_Music_Generation_AI`.
3. Configure the following settings:
   - **Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: CPU (free tier)
4. Render will read `runtime.txt`, assign a Python 3.11 container, build the virtual environment, and launch the Gunicorn server.

---

## 🤝 Contact & FAQ

- **How do I start training immediately?**
  We programmatically generate 5 sample MIDI files in `dataset/midi_files/` when the app runs for the first time. You can click **Start Training** right away!
- **How long does training take?**
  Training is heavy on CPU. We recommend setting epochs to `5` for a quick demonstration, and `50+` for full sequence learning on a machine equipped with a GPU.

For questions, feel free to open a GitHub issue or contact developers at `support@melodymindai.com`.
