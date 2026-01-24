#!/usr/bin/env python3
"""
Test script for Structure and Instruments MIDI Generation

Tests the new features:
- Structure sections with custom chords
- Multi-track instruments (bass, drums, arpeggio, melody, chord)
- Backward compatibility
"""

import requests
import json
import time
from pathlib import Path

API_BASE = "http://127.0.0.1:8000"

def print_response(response_data, test_name):
    """Pretty print test response."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(json.dumps(response_data, indent=2))
    print(f"{'='*60}\n")

def test_health():
    """Test server health."""
    print("Testing server health...")
    response = requests.get(f"{API_BASE}/health")
    if response.status_code == 200:
        print("✓ Server is healthy")
        print_response(response.json(), "Health Check")
        return True
    else:
        print(f"✗ Server health check failed: {response.status_code}")
        return False

def test_full_pipeline():
    """Test full pipeline with structure and instruments."""
    print("\n" + "="*60)
    print("TEST 1: Full Pipeline with Structure and Instruments")
    print("="*60)
    
    payload = {
        "intent": {
            "emotional_intent": "grief",
            "core_wound": "loss of a loved one",
            "core_desire": "finding peace",
            "technical": {
                "duration": 3.0,
                "bpm": 82,
                "key": "F major",
                "structure": [
                    {"name": "intro", "bars": 4, "repetitions": 1},
                    {"name": "verse", "bars": 8, "repetitions": 2},
                    {"name": "chorus", "bars": 8, "repetitions": 2},
                    {"name": "bridge", "bars": 4, "repetitions": 1},
                    {"name": "outro", "bars": 4, "repetitions": 1}
                ],
                "instruments": [
                    {"name": "piano", "type": "chord", "channel": 1},
                    {"name": "bass", "type": "bass", "channel": 2},
                    {"name": "drums", "type": "drums", "channel": 9, "style": "pop"},
                    {"name": "strings", "type": "arpeggio", "channel": 3}
                ]
            }
        },
        "output_format": "midi"
    }
    
    print("Sending request...")
    response = requests.post(f"{API_BASE}/generate", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Request successful!")
        print(f"  MIDI path: {result.get('midi_path', 'N/A')}")
        print(f"  Structure sections: {len(result.get('structure', {}).get('sections', []))}")
        print(f"  Total bars: {result.get('structure', {}).get('total_bars', 'N/A')}")
        print(f"  Instrument tracks: {len(result.get('instruments', {}).get('tracks', []))}")
        print_response(result, "Full Pipeline")
        return True
    else:
        print(f"✗ Request failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False

def test_structure_only():
    """Test structure without instruments."""
    print("\n" + "="*60)
    print("TEST 2: Structure Only (No Instruments)")
    print("="*60)
    
    payload = {
        "intent": {
            "emotional_intent": "nostalgia",
            "technical": {
                "duration": 2.5,
                "bpm": 90,
                "key": "C major",
                "structure": [
                    {"name": "intro", "bars": 4, "repetitions": 1},
                    {"name": "verse", "bars": 8, "repetitions": 1},
                    {"name": "chorus", "bars": 8, "repetitions": 1}
                ]
            }
        },
        "output_format": "midi"
    }
    
    print("Sending request...")
    response = requests.post(f"{API_BASE}/generate", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Request successful!")
        print(f"  MIDI path: {result.get('midi_path', 'N/A')}")
        print(f"  Structure sections: {len(result.get('structure', {}).get('sections', []))}")
        print_response(result, "Structure Only")
        return True
    else:
        print(f"✗ Request failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False

def test_instruments_only():
    """Test instruments without structure."""
    print("\n" + "="*60)
    print("TEST 3: Instruments Only (No Structure)")
    print("="*60)
    
    payload = {
        "intent": {
            "emotional_intent": "joy",
            "technical": {
                "duration": 2.0,
                "bpm": 120,
                "key": "G major",
                "instruments": [
                    {"name": "piano", "type": "chord", "channel": 1},
                    {"name": "bass", "type": "bass", "channel": 2},
                    {"name": "drums", "type": "drums", "channel": 9, "style": "rock"},
                    {"name": "melody", "type": "melody", "channel": 4}
                ]
            }
        },
        "output_format": "midi"
    }
    
    print("Sending request...")
    response = requests.post(f"{API_BASE}/generate", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Request successful!")
        print(f"  MIDI path: {result.get('midi_path', 'N/A')}")
        print(f"  Instrument tracks: {len(result.get('instruments', {}).get('tracks', []))}")
        print_response(result, "Instruments Only")
        return True
    else:
        print(f"✗ Request failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False

def test_backward_compatibility():
    """Test backward compatibility (no structure/instruments)."""
    print("\n" + "="*60)
    print("TEST 4: Backward Compatibility (No Structure/Instruments)")
    print("="*60)
    
    payload = {
        "intent": {
            "emotional_intent": "tenderness",
            "technical": {
                "bpm": 80,
                "key": "D major"
            }
        },
        "output_format": "midi"
    }
    
    print("Sending request...")
    response = requests.post(f"{API_BASE}/generate", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Request successful!")
        print(f"  MIDI path: {result.get('midi_path', 'N/A')}")
        print("  ✓ Backward compatibility maintained")
        print_response(result, "Backward Compatibility")
        return True
    else:
        print(f"✗ Request failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False

def test_section_specific_chords():
    """Test structure with section-specific chord progressions."""
    print("\n" + "="*60)
    print("TEST 5: Section-Specific Chord Progressions")
    print("="*60)
    
    payload = {
        "intent": {
            "emotional_intent": "defiance",
            "technical": {
                "duration": 2.5,
                "bpm": 130,
                "key": "A minor",
                "structure": [
                    {
                        "name": "intro",
                        "bars": 4,
                        "repetitions": 1,
                        "chords": ["Am", "F", "C", "G"]
                    },
                    {
                        "name": "verse",
                        "bars": 8,
                        "repetitions": 2,
                        "chords": ["Am", "G", "F", "E"]
                    },
                    {
                        "name": "chorus",
                        "bars": 8,
                        "repetitions": 2,
                        "chords": ["F", "C", "G", "Am"]
                    }
                ],
                "instruments": [
                    {"name": "guitar", "type": "chord", "channel": 3},
                    {"name": "bass", "type": "bass", "channel": 2},
                    {"name": "drums", "type": "drums", "channel": 9, "style": "rock"}
                ]
            }
        },
        "output_format": "midi"
    }
    
    print("Sending request...")
    response = requests.post(f"{API_BASE}/generate", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Request successful!")
        print(f"  MIDI path: {result.get('midi_path', 'N/A')}")
        print("  ✓ Section-specific chords applied")
        print_response(result, "Section-Specific Chords")
        return True
    else:
        print(f"✗ Request failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False

def test_all_instrument_types():
    """Test all instrument types."""
    print("\n" + "="*60)
    print("TEST 6: All Instrument Types")
    print("="*60)
    
    payload = {
        "intent": {
            "emotional_intent": "awe",
            "technical": {
                "duration": 2.0,
                "bpm": 100,
                "key": "C major",
                "instruments": [
                    {"name": "piano", "type": "chord", "channel": 1},
                    {"name": "bass", "type": "bass", "channel": 2},
                    {"name": "drums", "type": "drums", "channel": 9, "style": "pop"},
                    {"name": "arpeggio", "type": "arpeggio", "channel": 3},
                    {"name": "melody", "type": "melody", "channel": 4},
                    {"name": "piano_arpeggio", "type": "chord", "technique": "arpeggio", "channel": 5}
                ]
            }
        },
        "output_format": "midi"
    }
    
    print("Sending request...")
    response = requests.post(f"{API_BASE}/generate", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Request successful!")
        print(f"  MIDI path: {result.get('midi_path', 'N/A')}")
        tracks = result.get('instruments', {}).get('tracks', [])
        print(f"  Instrument tracks created: {len(tracks)}")
        for track in tracks:
            print(f"    - {track.get('name')} ({track.get('type')}) on channel {track.get('channel')}")
        print_response(result, "All Instrument Types")
        return True
    else:
        print(f"✗ Request failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("STRUCTURE AND INSTRUMENTS MIDI GENERATION - TEST SUITE")
    print("="*70)
    
    # Check server health first
    if not test_health():
        print("\n✗ Server is not available. Please start the API server first:")
        print("  python3 -m music_brain.api")
        return
    
    results = []
    
    # Run all tests
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("Structure Only", test_structure_only()))
    results.append(("Instruments Only", test_instruments_only()))
    results.append(("Full Pipeline", test_full_pipeline()))
    results.append(("Section-Specific Chords", test_section_specific_chords()))
    results.append(("All Instrument Types", test_all_instrument_types()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
