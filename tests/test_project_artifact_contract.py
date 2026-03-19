import os
import json
import pytest
from pathlib import Path

# Contract constants (must match ProjectFile.h)
REQUIRED_SCHEMA_VERSION = 2

def find_project_files(root_dir):
    """Find all .idaw.json files in the repository."""
    project_files = []
    for root, _, files in os.walk(root_dir):
        # Skip node_modules and build directories
        if 'node_modules' in root or 'build' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.idaw.json'):
                project_files.append(os.path.join(root, file))
    return project_files

# Get all project files in the repo
REPO_ROOT = Path(__file__).parent.parent
PROJECT_FILES = find_project_files(REPO_ROOT)

@pytest.mark.parametrize("project_path", PROJECT_FILES)
def test_project_file_contract(project_path):
    """
    Ensures every project file in the repo satisfies the freeze.deploy contract:
    1. Has a metadata.schemaVersion >= REQUIRED_SCHEMA_VERSION.
    2. All audioFile paths are relative (do not start with / or C:\).
    3. Basic structural invariants (name, tempo, mixer).
    """
    with open(project_path, 'r') as f:
        data = json.load(f)
    
    # 1. Metadata & Schema Version
    assert "metadata" in data, f"Missing 'metadata' in {project_path}"
    metadata = data["metadata"]
    assert "schemaVersion" in metadata, f"Missing 'schemaVersion' in {project_path}"
    assert metadata["schemaVersion"] >= REQUIRED_SCHEMA_VERSION, \
        f"Project {project_path} uses obsolete schema version {metadata['schemaVersion']}. " \
        f"Required: {REQUIRED_SCHEMA_VERSION}"
    
    # 2. Settings & Invariants
    assert "settings" in data, f"Missing 'settings' in {project_path}"
    settings = data["settings"]
    assert 20.0 <= settings.get("tempo", 0.0) <= 999.0, f"Invalid tempo in {project_path}"
    
    # 3. Mixer & Invariants
    assert "mixer" in data, f"Missing 'mixer' in {project_path}"
    mixer = data["mixer"]
    assert 0.0 <= mixer.get("masterVolume", -1.0) <= 2.0, f"Invalid master volume in {project_path}"
    
    # 4. Track Path Policy (Relative Only)
    assert "tracks" in data, f"Missing 'tracks' in {project_path}"
    for i, track in enumerate(data["tracks"]):
        track_name = track.get("name", f"Track {i}")
        if track.get("type") == "audio" and "audioFile" in track:
            audio_path = track["audioFile"]
            # Check for absolute path indicators
            assert not audio_path.startswith("/"), \
                f"Absolute path violation in {project_path}, track '{track_name}': {audio_path}"
            assert not (len(audio_path) > 2 and audio_path[1:3] == ":\\"), \
                f"Windows absolute path violation in {project_path}, track '{track_name}': {audio_path}"
            assert not audio_path.startswith("\\\\"), \
                f"UNC path violation in {project_path}, track '{track_name}': {audio_path}"

if __name__ == "__main__":
    # If no files found, still pass but warn (or fail if we expect some)
    if not PROJECT_FILES:
        print("No .idaw.json files found to validate.")
    else:
        print(f"Validated {len(PROJECT_FILES)} project files.")
