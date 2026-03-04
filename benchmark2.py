import time
import os
import sqlite3
import random
from music_brain.misc_code.audio_feel_extractor import init_database, get_connection, save_to_database, DB_PATH

# Remove db if exists
if DB_PATH.exists():
    os.remove(DB_PATH)

init_database()

# Create a fake analysis object with many genre matches and frequency balances
# to amplify the effect
analysis = {
    'metadata': {
        'name': 'test',
        'source_file': 'test.wav',
        'genre': 'test',
        'duration_seconds': 60.0,
        'sample_rate': 44100,
        'channels': 2,
        'estimated_bpm': 120.0,
        'estimated_key': 'C',
        'date_analyzed': '2023-01-01T00:00:00'
    },
    'transients': {},
    'dynamics': {},
    'spectrum': {},
    'frequency_balance': {
        'relative_to_mid_db': {f'band_{i}': 0.0 for i in range(1000)},
        'absolute_db': {f'band_{i}': 0.0 for i in range(1000)}
    },
    'stereo': {},
    'genre_matches': {f'genre_{i}': {'score': random.random()} for i in range(5000)}
}

print("Running baseline benchmark...")
start = time.time()
for _ in range(50):
    save_to_database(analysis)
end = time.time()
baseline_time = end - start

print(f"Baseline Time taken: {baseline_time:.4f} seconds")

# Save baseline to file
with open('benchmark_result.txt', 'w') as f:
    f.write(str(baseline_time))
