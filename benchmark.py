import time
import os
import sqlite3
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
    'genre_matches': {f'genre_{i}': {'score': 0.9} for i in range(1000)}
}

# The FREQ_BANDS are hardcoded in audio_feel_extractor, so our frequency_balance
# above will only insert the ones in FREQ_BANDS (which is like 8 bands).
# To properly test executemany vs execute on genre_matches, we have 1000 genre matches here.

start = time.time()
for _ in range(50):
    save_to_database(analysis)
end = time.time()

print(f"Time taken: {end - start:.4f} seconds")
