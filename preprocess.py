import os
import pickle
import numpy as np
from music21 import converter, note, chord, stream, meter, tempo, pitch
from keras.utils import to_categorical

# Configurable paths
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset', 'midi_files')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
NOTES_PKL = os.path.join(MODEL_DIR, 'notes.pkl')

def generate_dummy_midi_dataset():
    """
    Generates 5 simple sample MIDI files in the dataset folder if empty.
    This ensures that the app is functional out-of-the-box for users.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Check if there are already midi files
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.mid') or f.endswith('.midi')]
    if len(files) > 0:
        print(f"Dataset already contains {len(files)} MIDI files. Skipping dummy file generation.")
        return

    print("Dataset directory is empty. Generating sample MIDI files programmatically...")

    # Define some basic musical sequences
    # 1. C Major Scale and chords
    seq1 = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 
            'C4.E4.G4', 'F4.A4.C5', 'G4.B4.D5', 'C4.E4.G4',
            'C5', 'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4']

    # 2. A Minor Melodic Run
    seq2 = ['A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4', 'A4',
            'E4.G4.B4', 'A3.C4.E4', 'D4.F4.A4', 'E4.G#4.B4', 'A3.C4.E4',
            'A4', 'G4', 'F4', 'E4', 'D4', 'C4', 'B3', 'A3']

    # 3. I-V-vi-IV Chord progression in C Major (chords repeating with notes)
    seq3 = ['C4.E4.G4', 'G3.B3.D4', 'A3.C4.E4', 'F3.A3.C4',
            'C4', 'E4', 'G4', 'G3', 'B3', 'D4', 'A3', 'C4', 'E4', 'F3', 'A3', 'C4',
            'C4.E4.G4', 'G3.B3.D4', 'A3.C4.E4', 'F3.A3.C4']

    # 4. Pentatonic Theme (C Major Pentatonic: C, D, E, G, A)
    seq4 = ['C4', 'D4', 'E4', 'G4', 'A4', 'C5', 'D5', 'E5', 'G5', 'A5',
            'G5', 'E5', 'D5', 'C5', 'A4', 'G4', 'E4', 'D4', 'C4', 'G3.C4.E4']

    # 5. Happy Arpeggios and Chords
    seq5 = ['C4', 'E4', 'G4', 'C5', 'E4.G4.C5',
            'F4', 'A4', 'C5', 'F5', 'F4.A4.C5',
            'G4', 'B4', 'D5', 'G5', 'G4.B4.D5',
            'C4.E4.G4.C5']

    sequences = [seq1, seq2, seq3, seq4, seq5]
    
    # We repeat these patterns to create slightly longer files so that we have enough notes for sequence length 100
    for idx, seq in enumerate(sequences, 1):
        # Repeat the sequence 6 times to make sure each MIDI has > 100 notes
        extended_seq = seq * 6
        
        s = stream.Score()
        p = stream.Part()
        p.append(tempo.MetronomeMark(number=100 + (idx * 5))) # different tempos
        p.append(meter.TimeSignature('4/4'))
        
        for item in extended_seq:
            if '.' in item:
                # Chord
                chord_pitches = item.split('.')
                c = chord.Chord(chord_pitches)
                c.duration.quarterLength = 1.0
                p.append(c)
            else:
                # Single note
                n = note.Note(item)
                n.duration.quarterLength = 1.0
                p.append(n)
                
        s.append(p)
        filename = f"sample_song_{idx}.mid"
        filepath = os.path.join(DATASET_DIR, filename)
        s.write('midi', fp=filepath)
        print(f"Generated sample MIDI: {filepath}")

def get_notes():
    """
    Parses all MIDI files in dataset/midi_files/ and extracts notes/chords.
    Returns a flat list of all note strings.
    """
    notes = []
    
    # Ensure dataset is generated if empty
    generate_dummy_midi_dataset()
    
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.mid') or f.endswith('.midi')]
    
    if len(files) == 0:
        raise ValueError("No MIDI files found in the dataset directory, and sample generation failed.")
        
    print(f"Parsing {len(files)} MIDI files from {DATASET_DIR}...")
    
    for file in files:
        filepath = os.path.join(DATASET_DIR, file)
        try:
            # Parse MIDI file
            midi = converter.parse(filepath)
            print(f"Successfully parsed {file}")
            
            notes_to_parse = None
            
            # Try to get flat notes in a way compatible with all music21 versions (including v9+)
            try:
                if hasattr(midi, 'flatten'):
                    notes_to_parse = midi.flatten().notes
                elif hasattr(midi, 'flat'):
                    notes_to_parse = midi.flat.notes
                else:
                    notes_to_parse = midi.recurse().notes
            except Exception:
                try:
                    notes_to_parse = midi.recurse().notes
                except Exception:
                    try:
                        notes_to_parse = midi.notes
                    except Exception:
                        notes_to_parse = []
                
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    # Sort notes by pitch to ensure consistent representation (e.g. C4.E4 vs E4.C4)
                    sorted_pitches = sorted(element.notes, key=lambda x: x.pitch.ps)
                    notes.append('.'.join(str(n.pitch) for n in sorted_pitches))
        except Exception as e:
            print(f"Error parsing file {file}: {e}")
            continue
            
    print(f"Extraction complete. Total notes/chords extracted: {len(notes)}")
    return notes

def prepare_sequences(notes, n_vocab, note_to_int, sequence_length=100):
    """
    Prepares the input and output sequences for the neural network.
    """
    network_input = []
    network_output = []

    # Create input sequences and corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    if n_patterns == 0:
        raise ValueError(
            f"Not enough notes to create sequences of length {sequence_length}. "
            f"Total notes extracted: {len(notes)}. "
            f"Please decrease sequence length or add more MIDI files."
        )

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # Normalize input
    network_input = network_input / float(n_vocab)

    # One-hot encode the output
    network_output = to_categorical(network_output, num_classes=n_vocab)

    return network_input, network_output

def save_notes_mapping(notes):
    """
    Extracts unique pitches, creates mapping, and saves it.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    pitches = sorted(list(set(notes)))
    
    with open(NOTES_PKL, 'wb') as filepath:
        pickle.dump(pitches, filepath)
        
    print(f"Saved notes mapping (vocabulary size: {len(pitches)}) to {NOTES_PKL}")
    return pitches

if __name__ == '__main__':
    # Test preprocessing pipeline
    notes = get_notes()
    if notes:
        pitches = save_notes_mapping(notes)
        note_to_int = dict((note, number) for number, note in enumerate(pitches))
        X, y = prepare_sequences(notes, len(pitches), note_to_int)
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")
    else:
        print("Failed to preprocess notes.")
