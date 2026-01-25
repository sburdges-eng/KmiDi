#!/usr/bin/env python3
"""
Adapt import paths in test files from KmiDi structure to KmiDi-1 structure.
"""
import os
import re
from pathlib import Path
from typing import List

def adapt_imports(file_path: Path) -> bool:
    """Adapt imports in a test file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Common import adaptations
        replacements = [
            # Direct music_brain imports (should work as-is)
            # But check for any KmiDi-specific paths
            (r'from music_brain\.', 'from music_brain.'),
            (r'import music_brain', 'import music_brain'),
            
            # If there are any absolute path imports, we'll need to handle them
            # For now, most imports should work as-is since KmiDi-1 has the same structure
        ]
        
        # Check if file needs adaptation
        needs_adaptation = False
        for pattern, replacement in replacements:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                if content != original_content:
                    needs_adaptation = True
        
        # Write back if changed
        if needs_adaptation:
            file_path.write_text(content, encoding='utf-8')
            return True
        
        return False
    except Exception as e:
        print(f"Error adapting {file_path}: {e}")
        return False

def main():
    """Adapt all test files."""
    tests_dir = Path("/Users/seanburdges/KmiDi-1/tests/music_brain")
    
    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return
    
    adapted_count = 0
    for test_file in tests_dir.glob("test_*.py"):
        if adapt_imports(test_file):
            adapted_count += 1
            print(f"Adapted: {test_file.name}")
    
    print(f"\nAdapted {adapted_count} test files")
    print(f"Total test files: {len(list(tests_dir.glob('test_*.py')))}")

if __name__ == "__main__":
    main()
