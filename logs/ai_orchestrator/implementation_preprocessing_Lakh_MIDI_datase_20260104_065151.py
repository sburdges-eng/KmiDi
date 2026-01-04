"""
Generated Implementation: preprocessing Lakh MIDI dataset for melody generation - data loading and feature extraction
Based on deep research conducted 2026-01-04T06:51:51.970379
"""

import pretty_midi
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class MIDIProcessor:
    """
    A class to preprocess MIDI files for music generation models.
    
    This class handles loading MIDI files, extracting features, and preparing
    data for training models such as LSTM and MLP.
    """

    def __init__(self, midi_files):
        """
        Initializes the MIDIProcessor with a list of MIDI file paths.
        
        Args:
            midi_files (list): List of paths to MIDI files.
        """
        self.midi_files = midi_files

    def load_midi(self, file_path):
        """
        Loads a MIDI file and returns a PrettyMIDI object.
        
        Args:
            file_path (str): Path to the MIDI file.
        
        Returns:
            pretty_midi.PrettyMIDI: Parsed MIDI file.
        """
        return pretty_midi.PrettyMIDI(file_path)

    def extract_features(self, midi_data):
        """
        Extracts features such as pitch, velocity, and timing from MIDI data.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): Parsed MIDI data.
        
        Returns:
            dict: Dictionary containing extracted features.
        """
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append((note.start, note.pitch, note.velocity, note.end - note.start))
        
        # Sort notes by start time
        notes.sort(key=lambda x: x[0])
        
        # Extract features
        start_times = [note[0] for note in notes]
        pitches = [note[1] for note in notes]
        velocities = [note[2] for note in notes]
        durations = [note[3] for note in notes]

        return {
            'start_times': start_times,
            'pitches': pitches,
            'velocities': velocities,
            'durations': durations
        }

    def preprocess_data(self):
        """
        Preprocesses all MIDI files and returns padded sequences for model training.
        
        Returns:
            tuple: Tuple containing padded sequences for pitches, velocities, and durations.
        """
        all_pitches = []
        all_velocities = []
        all_durations = []

        for file_path in self.midi_files:
            midi_data = self.load_midi(file_path)
            features = self.extract_features(midi_data)
            
            all_pitches.append(features['pitches'])
            all_velocities.append(features['velocities'])
            all_durations.append(features['durations'])

        # Pad sequences to a fixed length
        max_length = 200  # Based on research findings
        padded_pitches = pad_sequences(all_pitches, maxlen=max_length, padding='post')
        padded_velocities = pad_sequences(all_velocities, maxlen=max_length, padding='post')
        padded_durations = pad_sequences(all_durations, maxlen=max_length, padding='post')

        return padded_pitches, padded_velocities, padded_durations

# Example usage
if __name__ == "__main__":
    midi_files = ['path/to/midi1.mid', 'path/to/midi2.mid']  # Replace with actual paths
    processor = MIDIProcessor(midi_files)
    pitches, velocities, durations = processor.preprocess_data()

    # Example of creating a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((pitches, velocities, durations))
    dataset = dataset.batch(32).shuffle(buffer_size=1000)

    # Now, this dataset can be used to train models like melody_transformer or groove_predictor
