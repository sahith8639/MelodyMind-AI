import os
import json
import time
import threading
from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

from preprocess import get_notes, DATASET_DIR, MODEL_DIR, NOTES_PKL
from train_model import train_network, HISTORY_PATH, MODEL_PATH
from generate_music import generate_music, GENERATED_DIR

app = Flask(__name__)
app.config['SECRET_KEY'] = 'melodymind_ai_secret_key_123987'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB Limit

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

# Allowed extensions for MIDI files
ALLOWED_EXTENSIONS = {'mid', 'midi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global training thread lock/reference
training_lock = threading.Lock()
training_thread = None

def get_training_status():
    """
    Reads the current training status from history JSON file.
    """
    if not os.path.exists(HISTORY_PATH):
        return {"status": "idle"}
    try:
        with open(HISTORY_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {"status": "idle"}

def run_training_async(epochs, batch_size):
    """
    Wrapper function run in a separate thread to execute the training.
    """
    try:
        train_network(epochs=epochs, batch_size=batch_size)
    except Exception as e:
        print(f"Background training failed: {e}")

# ==========================================
# FLASK ROUTING
# ==========================================

@app.route('/')
def home():
    # 1. Count MIDI files in dataset
    midi_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.mid') or f.endswith('.midi')]
    midi_count = len(midi_files)
    
    # 2. Get status from training history
    history = get_training_status()
    status_str = history.get('status', 'Not Trained')
    epochs_completed = history.get('current_epoch', 0)
    total_epochs = history.get('total_epochs', 0)
    
    accuracy = history.get('accuracy', 0.0)
    loss = history.get('loss', 0.0)
    
    # 3. Read total notes if notes.pkl exists
    import pickle as _pickle
    total_notes_learned = 0
    if os.path.exists(NOTES_PKL):
        try:
            with open(NOTES_PKL, 'rb') as pf:
                pitches = _pickle.load(pf)
                total_notes_learned = len(pitches)
        except Exception:
            pass
            
    # 4. Count generated songs
    generated_songs = [f for f in os.listdir(GENERATED_DIR) if f.endswith('.mid')]
    generated_count = len(generated_songs)

    dashboard_stats = {
        "midi_count": midi_count,
        "notes_learned": total_notes_learned,
        "accuracy": f"{accuracy * 100:.1f}%" if accuracy > 0 else "N/A",
        "loss": f"{loss:.4f}" if loss > 0 else "N/A",
        "epochs_completed": f"{epochs_completed}/{total_epochs}" if total_epochs > 0 else "N/A",
        "generated_count": generated_count,
        "status": status_str
    }
    
    return render_template('index.html', stats=dashboard_stats)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/train')
def train():
    # List MIDI files in dataset
    midi_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.mid') or f.endswith('.midi')]
    history = get_training_status()
    
    # Check if model is trained
    model_exists = os.path.exists(MODEL_PATH)
    
    return render_template('train.html', midi_files=midi_files, history=history, model_exists=model_exists)

@app.route('/upload-midi', methods=['POST'])
def upload_midi():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Ensure name uniqueness or overwrite is fine
        filepath = os.path.join(DATASET_DIR, filename)
        file.save(filepath)
        return jsonify({
            "message": f"Successfully uploaded {filename}",
            "filename": filename
        }), 200
        
    return jsonify({"error": "Invalid file type. Only .mid and .midi files are allowed."}), 400

@app.route('/start-training', methods=['POST'])
def start_training():
    global training_thread
    
    # Check lock
    with training_lock:
        history = get_training_status()
        active_states = ["preprocessing", "preparing_sequences", "training"]
        if history.get("status") in active_states:
            return jsonify({"error": "Training is already in progress."}), 400
            
        # Parse arguments
        try:
            data = request.get_json() or {}
            epochs = int(data.get('epochs', 50))
            batch_size = int(data.get('batch_size', 64))
        except Exception:
            epochs = 50
            batch_size = 64
            
        # Validate
        if epochs <= 0 or epochs > 200:
            epochs = 50
        if batch_size <= 0 or batch_size > 256:
            batch_size = 64
            
        # Spin thread
        training_thread = threading.Thread(
            target=run_training_async,
            args=(epochs, batch_size)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "message": "Training started in background.",
            "epochs": epochs,
            "batch_size": batch_size
        }), 200

@app.route('/status')
def status():
    history = get_training_status()
    # Check if model exists
    history["model_exists"] = os.path.exists(MODEL_PATH)
    return jsonify(history)

@app.route('/generate')
def generate():
    model_exists = os.path.exists(MODEL_PATH)
    notes_exist = os.path.exists(NOTES_PKL)
    
    # Fetch list of existing generated files
    files = []
    if os.path.exists(GENERATED_DIR):
        # List files and sort by modification time (newest first)
        midi_files = [f for f in os.listdir(GENERATED_DIR) if f.endswith('.mid')]
        
        for f in midi_files:
            filepath = os.path.join(GENERATED_DIR, f)
            stat = os.stat(filepath)
            # Find matching WAV if exists
            wav_name = f.rsplit('.', 1)[0] + '.wav'
            wav_exists = os.path.exists(os.path.join(GENERATED_DIR, wav_name))
            
            files.append({
                "midi_file": f,
                "wav_file": wav_name if wav_exists else None,
                "created_at": stat.st_mtime,
                "formatted_date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
            })
            
        files.sort(key=lambda x: x['created_at'], reverse=True)
        
    return render_template('generate.html', model_exists=model_exists and notes_exist, history=files)

@app.route('/generate-music', methods=['POST'])
def generate_music_endpoint():
    # Ensure model exists
    if not os.path.exists(MODEL_PATH) or not os.path.exists(NOTES_PKL):
        return jsonify({"error": "Model is not trained. Please train the model before generating music."}), 400
        
    try:
        data = request.get_json() or {}
        num_notes = int(data.get('num_notes', 500))
        temperature = float(data.get('temperature', 1.0))
        output_name = secure_filename(data.get('output_name', 'generated_song'))
        
        if not output_name:
            output_name = 'generated_song'
            
        # Clean extensions if user typed them
        if output_name.lower().endswith('.mid') or output_name.lower().endswith('.midi'):
            output_name = output_name.rsplit('.', 1)[0]
            
        # Generate music (blocks momentarily, which is fine since it takes ~10-20 seconds on modern CPU for 500 notes)
        # We run it synchronously so the AJAX call gets the final result when complete
        result = generate_music(num_notes=num_notes, temperature=temperature, output_name=output_name)
        
        return jsonify({
            "message": "Music generated successfully!",
            "midi_url": url_for('download_file', filename=result['midi_file']),
            "wav_url": url_for('download_file', filename=result['wav_file']) if result['wav_file'] else None,
            "midi_file": result['midi_file'],
            "wav_file": result['wav_file'],
            "latest_midi_url": url_for('download_file', filename=result['latest_midi_file']),
            "latest_wav_url": url_for('download_file', filename=result['latest_wav_file']) if result['latest_wav_file'] else None,
            "temperature": temperature,
            "num_notes": num_notes
        }), 200
        
    except Exception as e:
        print(f"Error during music generation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    # Ensure secure filename to avoid directory traversal
    filename = secure_filename(filename)
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Flask Dev Server
    app.run(host='0.0.0.0', port=5000, debug=True)
