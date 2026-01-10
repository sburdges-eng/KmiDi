#!/usr/bin/env python3
"""
Test FastAPI service endpoints locally.

Tests:
- /health endpoint
- /emotions endpoint
- /generate endpoint
"""

import sys
import time
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_health_endpoint(base_url="http://localhost:8000"):
    """Test health check endpoint."""
    print("=" * 70)
    print("Testing /health endpoint")
    print("=" * 70)
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {response.status_code}")
        print(f"  Response: {data}")
        
        # Accept both 'healthy' and 'ok' status
        assert data["status"] in ["healthy", "ok"], f"Expected 'healthy' or 'ok', got '{data['status']}'"
        assert "version" in data, "Missing version field"
        # Services field is optional for compatibility
        
        print(f"  ✓ Health check passed")
        return True
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Connection failed - is the server running?")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_emotions_endpoint(base_url="http://localhost:8000"):
    """Test emotions listing endpoint."""
    print("\n" + "=" * 70)
    print("Testing /emotions endpoint")
    print("=" * 70)
    
    try:
        response = requests.get(f"{base_url}/emotions", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {response.status_code}")
        print(f"  Response: {data}")
        
        # Handle both dict and list responses
        if isinstance(data, list):
            emotions = data
            count = len(emotions)
        else:
            assert "emotions" in data, "Missing emotions field"
            emotions = data["emotions"]
            count = data.get("count", len(emotions))
        
        assert isinstance(emotions, list), "Emotions should be a list"
        
        print(f"  ✓ Found {count} emotions")
        if emotions:
            print(f"  Sample emotions: {', '.join(emotions[:5])}")
        
        return True
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Connection failed - is the server running?")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_generate_endpoint(base_url="http://localhost:8000"):
    """Test music generation endpoint."""
    print("\n" + "=" * 70)
    print("Testing /generate endpoint")
    print("=" * 70)
    
    try:
        payload = {
            "intent": {
                "emotional_intent": "I'm feeling sad and melancholic",
                "core_wound": None,
                "core_desire": None,
                "technical": {
                    "key": None,
                    "bpm": 72,
                    "progression": None,
                    "genre": None
                }
            },
            "output_format": "midi"
        }
        
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            timeout=30  # Generation may take time
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {response.status_code}")
        print(f"  Response keys: {list(data.keys())}")
        
        assert "status" in data, "Missing status field"
        assert data["status"] == "success", f"Expected 'success', got '{data['status']}'"
        
        print(f"  ✓ Generation successful")
        if "result" in data:
            result = data["result"]
            if isinstance(result, dict) and "musical_params" in result:
                params = result["musical_params"]
                print(f"  Tempo: {params.get('tempo_suggested', 'N/A')} BPM")
        
        return True
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Connection failed - is the server running?")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                print(f"    Details: {error_data}")
            except:
                print(f"    Response: {e.response.text[:200]}")
        return False


def main():
    """Run all endpoint tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FastAPI endpoints")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for API")
    parser.add_argument("--wait", action="store_true", help="Wait for server to be ready")
    args = parser.parse_args()
    
    if args.wait:
        print("Waiting for server to be ready...")
        max_attempts = 30
        for i in range(max_attempts):
            try:
                response = requests.get(f"{args.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print("✓ Server is ready!")
                    break
            except:
                pass
            time.sleep(1)
            if i == max_attempts - 1:
                print("✗ Server did not become ready in time")
                return 1
    
    results = []
    results.append(("Health", test_health_endpoint(args.base_url)))
    results.append(("Emotions", test_emotions_endpoint(args.base_url)))
    results.append(("Generate", test_generate_endpoint(args.base_url)))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}  {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All endpoint tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

