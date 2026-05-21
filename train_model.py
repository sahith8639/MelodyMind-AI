import os
import json
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam

from preprocess import get_notes, prepare_sequences, save_notes_mapping

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'music_model.h5')
HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.json')
NOTES_PKL = os.path.join(MODEL_DIR, 'notes.pkl')

class TrainingProgressCallback(Callback):
    """
    Custom Keras Callback to log training metrics after each epoch
    into a JSON file, which the Flask server polls to display progress.
    """
    def __init__(self, history_file, total_epochs):
        super().__init__()
        self.history_file = history_file
        self.total_epochs = total_epochs
        self.start_time = time.time()
        
    def on_train_begin(self, logs=None):
        initial_status = {
            "status": "training",
            "current_epoch": 0,
            "total_epochs": self.total_epochs,
            "loss": 0.0,
            "accuracy": 0.0,
            "elapsed_time": 0.0,
            "last_updated": time.time()
        }
        self.write_history(initial_status)
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self.start_time
        status = {
            "status": "training",
            "current_epoch": epoch + 1,
            "total_epochs": self.total_epochs,
            "loss": float(logs.get('loss', 0.0)),
            "accuracy": float(logs.get('accuracy', 0.0)),
            "elapsed_time": float(elapsed),
            "last_updated": time.time()
        }
        self.write_history(status)
        
    def write_history(self, data):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error writing training progress: {e}")

def build_model(input_shape, output_size):
    """
    Builds the LSTM network model.
    """
    model = Sequential()
    
    # First LSTM layer with 512 units, return_sequences=True
    model.add(Input(shape=(input_shape[1], input_shape[2])))
    model.add(LSTM(
        512,
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    
    # Second LSTM layer with 512 units, return_sequences=False
    model.add(LSTM(512, return_sequences=False))
    
    # Dense layer with 256 units
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    
    # Softmax output layer
    model.add(Dense(output_size, activation='softmax'))
    
    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

def train_network(epochs=50, batch_size=64):
    """
    Runs the complete model training pipeline.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Initialize history file to "preprocessing" state
    initial_progress = {
        "status": "preprocessing",
        "current_epoch": 0,
        "total_epochs": epochs,
        "loss": 0.0,
        "accuracy": 0.0,
        "elapsed_time": 0.0,
        "last_updated": time.time()
    }
    with open(HISTORY_PATH, 'w') as f:
        json.dump(initial_progress, f, indent=4)
        
    try:
        # 1. Get notes and chords
        notes = get_notes()
        n_vocab = len(set(notes))
        
        # 2. Save note mappings
        pitches = save_notes_mapping(notes)
        note_to_int = dict((note, number) for number, note in enumerate(pitches))
        
        # Write stats to history file for display
        initial_progress["status"] = "preparing_sequences"
        initial_progress["notes_count"] = len(notes)
        initial_progress["vocab_size"] = n_vocab
        with open(HISTORY_PATH, 'w') as f:
            json.dump(initial_progress, f, indent=4)
            
        # 3. Format inputs and targets
        X, y = prepare_sequences(notes, n_vocab, note_to_int)
        
        # 4. Build the LSTM network
        model = build_model(X.shape, n_vocab)
        model.summary()
        
        # 5. Set up callbacks
        checkpoint = ModelCheckpoint(
            MODEL_PATH,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        
        progress_callback = TrainingProgressCallback(HISTORY_PATH, epochs)
        
        # 6. Start model training
        print(f"Starting model training for {epochs} epochs...")
        model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, progress_callback],
            verbose=1
        )
        
        # 7. Finalize history to completed status
        final_history = {
            "status": "completed",
            "current_epoch": epochs,
            "total_epochs": epochs,
            "notes_count": len(notes),
            "vocab_size": n_vocab,
            "last_updated": time.time()
        }
        
        # Retrieve final loss and accuracy if training completed successfully
        if len(model.history.history.get('loss', [])) > 0:
            final_history["loss"] = float(model.history.history['loss'][-1])
            final_history["accuracy"] = float(model.history.history['accuracy'][-1])
            
        with open(HISTORY_PATH, 'w') as f:
            json.dump(final_history, f, indent=4)
            
        print("Training successfully completed.")
        
    except Exception as e:
        print(f"Error during training: {e}")
        failed_status = {
            "status": "failed",
            "error_message": str(e),
            "last_updated": time.time()
        }
        with open(HISTORY_PATH, 'w') as f:
            json.dump(failed_status, f, indent=4)
        raise e

if __name__ == '__main__':
    # Train model with shorter default epochs for quick testing if run directly
    # Can run full 50 epochs when requested
    train_network(epochs=5, batch_size=64)
