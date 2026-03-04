with open('music_brain/misc_code/audio_feel_extractor.py', 'r') as f:
    content = f.read()

search1 = """    # Frequency balance
    freq = analysis.get('frequency_balance', {})
    relative = freq.get('relative_to_mid_db', {})
    absolute = freq.get('absolute_db', {})
    for band in FREQ_BANDS.keys():
        if band in relative:
            cursor.execute('''
                INSERT INTO frequency_balance
                (analysis_id, band_name, energy_db, relative_to_mid_db)
                VALUES (?, ?, ?, ?)
            ''', (analysis_id, band, absolute.get(band, 0), relative.get(band, 0)))"""

replace1 = """    # Frequency balance
    freq = analysis.get('frequency_balance', {})
    relative = freq.get('relative_to_mid_db', {})
    absolute = freq.get('absolute_db', {})
    freq_data = []
    for band in FREQ_BANDS.keys():
        if band in relative:
            freq_data.append((analysis_id, band, absolute.get(band, 0), relative.get(band, 0)))

    if freq_data:
        cursor.executemany('''
            INSERT INTO frequency_balance
            (analysis_id, band_name, energy_db, relative_to_mid_db)
            VALUES (?, ?, ?, ?)
        ''', freq_data)"""

content = content.replace(search1, replace1)

search2 = """    # Genre matches
    for genre, match_data in analysis.get('genre_matches', {}).items():
        cursor.execute('''
            INSERT INTO genre_matches
            (analysis_id, genre, match_score)
            VALUES (?, ?, ?)
        ''', (analysis_id, genre, match_data['score']))"""

replace2 = """    # Genre matches
    genre_data = []
    for genre, match_data in analysis.get('genre_matches', {}).items():
        genre_data.append((analysis_id, genre, match_data['score']))

    if genre_data:
        cursor.executemany('''
            INSERT INTO genre_matches
            (analysis_id, genre, match_score)
            VALUES (?, ?, ?)
        ''', genre_data)"""

content = content.replace(search2, replace2)

with open('music_brain/misc_code/audio_feel_extractor.py', 'w') as f:
    f.write(content)
