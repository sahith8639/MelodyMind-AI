import os
import pickle
import numpy as np
import tensorflow as tf
import wave
import struct
import time
from music21 import stream, note, chord, tempo, meter
import pretty_midi

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'music_model.h5')
NOTES_PKL = os.path.join(MODEL_DIR, 'notes.pkl')
GENERATED_DIR = os.path.join(os.path.dirname(__file__), 'generated')

def save_numpy_to_wav(audio_data, filepath, sample_rate=22050):
    """
    Saves a float numpy array as a 16-bit PCM mono WAV file using Python's 
    built-in wave and struct modules. Highly portable and dependency-free.
    """
    # Normalize audio to prevent clipping and maximize volume
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
        
    # Scale to 16-bit signed integer range (-32768 to 32767)
    audio_ints = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
    
    # Open WAV file for writing
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)       # Mono
        wf.setsampwidth(2)      # 16-bit (2 bytes)
        wf.setframerate(sample_rate)
        
        # Pack short ints into binary little-endian format
        binary_data = struct.pack('<' + 'h' * len(audio_ints), *audio_ints)
        wf.writeframes(binary_data)
        
    print(f"Saved synthesized audio WAV to: {filepath}")

def synthesize_midi_to_wav(midi_path, wav_path, sample_rate=22050):
    """
    Uses pretty_midi's built-in sine wave synthesizer to convert
    a MIDI file into a WAV file for browser playback.
    """
    print(f"Synthesizing {midi_path} to WAV...")
    try:
        # Load MIDI data
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        # Synthesize audio using built-in wave synth (sine by default)
        audio_data = midi_data.synthesize(fs=sample_rate)
        
        # Save synthesized waveform to WAV
        save_numpy_to_wav(audio_data, wav_path, sample_rate)
        return True
    except Exception as e:
        print(f"Failed to synthesize MIDI to WAV: {e}")
        return False

def generate_music(num_notes=500, temperature=1.0, output_name="generated_song"):
    """
    Generates a MIDI file and its synthesized WAV file using the trained model.
    """
    os.makedirs(GENERATED_DIR, exist_ok=True)
    
    # 1. Load notes mapping
    if not os.path.exists(NOTES_PKL):
        raise FileNotFoundError(f"Notes mapping file not found at {NOTES_PKL}. Please train the model first.")
        
    with open(NOTES_PKL, 'rb') as f:
        pitches = pickle.load(f)
        
    n_vocab = len(pitches)
    note_to_int = dict((note, number) for number, note in enumerate(pitches))
    int_to_note = dict((number, note) for number, note in enumerate(pitches))
    
    # 2. Load trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please train the model first.")
        
    print(f"Loading trained AI model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 3. Create a seed sequence
    # To get a seed, we parse a random file in the dataset or make one up if not available
    # For robust seed selection, we'll try to extract notes from the dataset
    from preprocess import get_notes
    try:
        dataset_notes = get_notes()
        if len(dataset_notes) >= 100:
            # Pick a random 100-note window from the dataset notes
            start_idx = np.random.randint(0, len(dataset_notes) - 100)
            seed_sequence = dataset_notes[start_idx:start_idx + 100]
        else:
            raise ValueError("Dataset too short for seed sequence.")
    except Exception as e:
        print(f"Could not extract seed from dataset: {e}. Using a fallback procedural seed.")
        # Fallback procedural seed if no dataset is parsed
        fallback_notes = ['C4', 'E4', 'G4', 'C5', 'B4', 'G4', 'E4', 'C4'] * 13
        seed_sequence = fallback_notes[:100]

    # Convert notes to integers
    # Make sure notes in seed_sequence are in our vocabulary, otherwise replace with a random note
    seed_ints = []
    for n in seed_sequence:
        if n in note_to_int:
            seed_ints.append(note_to_int[n])
        else:
            seed_ints.append(np.random.choice(n_vocab))
            
    print(f"Seed sequence established (length: {len(seed_ints)})")

    # 4. Generate sequences
    prediction_output = []
    current_sequence = seed_ints.copy()
    
    print(f"Generating {num_notes} notes/chords with temperature={temperature}...")
    
    for note_index in range(num_notes):
        # Format input sequence
        prediction_input = np.reshape(current_sequence, (1, len(current_sequence), 1))
        prediction_input = prediction_input / float(n_vocab)
        
        # Predict probability distribution
        predictions = model.predict(prediction_input, verbose=0)[0]
        
        # Apply temperature-based sampling
        if temperature <= 0.0:
            # Argmax (deterministic)
            prediction_idx = np.argmax(predictions)
        else:
            # Avoid division-by-zero or extreme errors, clip predictions
            predictions = np.clip(predictions, 1e-7, 1.0)
            log_predictions = np.log(predictions) / temperature
            exp_predictions = np.exp(log_predictions)
            softmax_predictions = exp_predictions / np.sum(exp_predictions)
            
            # Draw sample from distribution
            prediction_idx = np.random.choice(n_vocab, p=softmax_predictions)
            
        note_str = int_to_note[prediction_idx]
        prediction_output.append(note_str)
        
        # Slide window: append predicted note and drop the first one
        current_sequence.append(prediction_idx)
        current_sequence = current_sequence[1:]

    # 5. Convert generated sequence back to music21 Stream
    print("Converting generated sequence to MIDI file...")
    s = stream.Score()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=100))
    p.append(meter.TimeSignature('4/4'))
    
    # We'll use a fixed short duration (e.g., 0.5 = eighth note) for a steady rhythmic flow
    # Varying the rhythm slightly can make it sound less robotic, but consistent is a solid baseline.
    for pattern in prediction_output:
        # Check if it is a chord
        if '.' in pattern:
            chord_pitches = pattern.split('.')
            notes_in_chord = []
            for current_note in chord_pitches:
                new_note = note.Note(current_note)
                notes_in_chord.append(new_note)
            new_chord = chord.Chord(notes_in_chord)
            new_chord.duration.quarterLength = 0.5 # Play eighth notes
            p.append(new_chord)
        else:
            # Single note
            new_note = note.Note(pattern)
            new_note.duration.quarterLength = 0.5
            p.append(new_note)
            
    s.append(p)
    
    # Save filenames
    timestamp = int(time.time())
    midi_filename = f"{output_name}_{timestamp}.mid"
    wav_filename = f"{output_name}_{timestamp}.wav"
    
    midi_path = os.path.join(GENERATED_DIR, midi_filename)
    wav_path = os.path.join(GENERATED_DIR, wav_filename)
    
    # Save specific timestamp file
    s.write('midi', fp=midi_path)
    print(f"Generated MIDI saved to {midi_path}")
    
    # Also save as the default "latest" file
    latest_midi_path = os.path.join(GENERATED_DIR, f"{output_name}.mid")
    latest_wav_path = os.path.join(GENERATED_DIR, f"{output_name}.wav")
    
    # Write to latest file as well
    s.write('midi', fp=latest_midi_path)
    
    # 6. Synthesize to WAV
    success = synthesize_midi_to_wav(midi_path, wav_path)
    if success:
        # Copy to latest WAV path
        import shutil
        shutil.copy(wav_path, latest_wav_path)
        
    return {
        "midi_file": midi_filename,
        "wav_file": wav_filename if success else None,
        "midi_path": midi_path,
        "wav_path": wav_path if success else None,
        "latest_midi_file": f"{output_name}.mid",
        "latest_wav_file": f"{output_name}.wav" if success else None,
        "num_notes": num_notes,
        "temperature": temperature,
        "timestamp": timestamp
    }

if __name__ == '__main__':
    # Test generation script
    try:
        stats = generate_music(num_notes=100, temperature=1.0)
        print("Generated files successfully:")
        print(stats)
    except Exception as e:
        print(f"Failed to generate music: {e}")
