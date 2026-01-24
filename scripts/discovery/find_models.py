#!/usr/bin/env python3
"""
Find and catalog all ML model files across the MacBook Pro.
Analyzes model files, checkpoints, and training artifacts.
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

# Model file extensions
MODEL_EXTENSIONS = {
    'pytorch': {'.pt', '.pth'},
    'tensorflow': {'.pb', '.h5', '.ckpt', '.ckpt.index'},
    'onnx': {'.onnx'},
    'tflite': {'.tflite'},
    'coreml': {'.mlmodel'},
    'pickle': {'.pkl', '.pickle'},
    'safetensors': {'.safetensors'},
}

ALL_MODEL_EXTENSIONS = set()
for exts in MODEL_EXTENSIONS.values():
    ALL_MODEL_EXTENSIONS.update(exts)

# Training artifact patterns
CHECKPOINT_PATTERNS = [
    r'epoch_\d+',
    r'checkpoint_\d+',
    r'step_\d+',
    r'best\.pt',
    r'best_model\.pt',
    r'final\.pt',
    r'last\.pt',
]

EXCLUDE_DIRS = {
    'Library', 'System', 'Applications', 'node_modules', '.git', '__pycache__',
    '.pytest_cache', '.mypy_cache', '.venv', 'venv', 'env', 'build', 'dist'
}

def should_exclude(path: Path) -> bool:
    """Check if path should be excluded."""
    parts = path.parts
    for exclude in EXCLUDE_DIRS:
        if exclude in parts:
            return True
    return False

def get_model_type(file_path: Path) -> str:
    """Determine model type from extension."""
    ext = file_path.suffix.lower()
    for model_type, exts in MODEL_EXTENSIONS.items():
        if ext in exts:
            return model_type
    return 'unknown'

def is_checkpoint(file_path: Path) -> bool:
    """Check if file is a training checkpoint."""
    import re
    name = file_path.stem.lower()
    for pattern in CHECKPOINT_PATTERNS:
        if re.search(pattern, name):
            return True
    return False

def find_config_files(model_path: Path) -> List[str]:
    """Find associated configuration files."""
    config_files = []
    parent = model_path.parent
    
    # Common config file names
    config_names = [
        'config.json', 'config.yaml', 'config.yml',
        'hyperparameters.json', 'params.json',
        'model_config.json', 'training_config.json',
        model_path.stem + '_config.json',
        model_path.stem + '.json',
    ]
    
    for config_name in config_names:
        config_path = parent / config_name
        if config_path.exists():
            config_files.append(str(config_path))
    
    return config_files

def analyze_model_file(file_path: Path) -> Dict:
    """Analyze a model file."""
    try:
        stat = file_path.stat()
        model_type = get_model_type(file_path)
        is_ckpt = is_checkpoint(file_path)
        config_files = find_config_files(file_path)
        
        file_info = {
            'path': str(file_path),
            'name': file_path.name,
            'type': model_type,
            'size': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'size_gb': round(stat.st_size / (1024 * 1024 * 1024), 2),
            'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'directory': str(file_path.parent),
            'is_checkpoint': is_ckpt,
            'config_files': config_files,
        }
        
        return file_info
    
    except Exception as e:
        return None

def find_model_files(root: Path = Path('/Users/seanburdges'), max_files: int | None = None) -> List[Dict]:
    """Find all model files. max_files=None means no limit."""
    model_files = []
    
    print(f"Scanning for model files starting from: {root}" + (f" (max {max_files})" if max_files else " (no limit)"))
    
    for file_path in root.rglob('*'):
        if max_files is not None and len(model_files) >= max_files:
            print(f"Reached limit of {max_files} files")
            break
        
        if should_exclude(file_path):
            continue
        
        if file_path.is_file():
            # Check extension
            if file_path.suffix.lower() in ALL_MODEL_EXTENSIONS:
                file_info = analyze_model_file(file_path)
                if file_info:
                    model_files.append(file_info)
                    
                    if len(model_files) % 100 == 0 and len(model_files) > 0:
                        print(f"Found {len(model_files)} model files...")
    
    return model_files

def categorize_files(model_files: List[Dict]) -> Dict:
    """Categorize model files."""
    categories = {
        'by_type': defaultdict(int),
        'checkpoints': [],
        'large_models': [],  # > 100MB
        'very_large_models': [],  # > 1GB
        'recent_files': [],
        'with_configs': [],
        'total_size_gb': 0,
    }
    
    cutoff_date = datetime.now().timestamp() - (30 * 24 * 60 * 60)
    
    for file_info in model_files:
        categories['by_type'][file_info['type']] += 1
        categories['total_size_gb'] += file_info['size_gb']
        
        if file_info['is_checkpoint']:
            categories['checkpoints'].append(file_info['path'])
        
        if file_info['size_mb'] > 100:
            categories['large_models'].append({
                'path': file_info['path'],
                'size_mb': file_info['size_mb']
            })
        
        if file_info['size_gb'] > 1:
            categories['very_large_models'].append({
                'path': file_info['path'],
                'size_gb': file_info['size_gb']
            })
        
        try:
            mod_time = datetime.fromisoformat(file_info['last_modified']).timestamp()
            if mod_time > cutoff_date:
                categories['recent_files'].append(file_info['path'])
        except:
            pass
        
        if file_info['config_files']:
            categories['with_configs'].append({
                'path': file_info['path'],
                'configs': file_info['config_files']
            })
    
    categories['total_size_gb'] = round(categories['total_size_gb'], 2)
    
    return categories

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Find and catalog ML model files")
    parser.add_argument("--max-files", type=int, default=None, help="Cap number of files (default: no limit)")
    parser.add_argument("--root", type=Path, default=Path("/Users/seanburdges"), help="Root directory to scan")
    args = parser.parse_args()

    print("Finding all model files...")
    print("=" * 60)
    
    model_files = find_model_files(root=args.root, max_files=args.max_files)
    categories = categorize_files(model_files)
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'docs' / 'DISCOVERY'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / 'model_files_inventory.json'
    with open(json_path, 'w') as f:
        json.dump({
            'scan_timestamp': datetime.now().isoformat(),
            'total_files': len(model_files),
            'files': model_files,
            'categories': categories
        }, f, indent=2)
    print(f"\nSaved JSON results to: {json_path}")
    
    # Save Markdown report
    md_path = output_dir / 'model_files_inventory.md'
    with open(md_path, 'w') as f:
        f.write("# Model Files Inventory\n\n")
        f.write(f"**Scan Date**: {datetime.now().isoformat()}\n\n")
        f.write(f"**Total Model Files Found**: {len(model_files)}\n")
        f.write(f"**Total Size**: {categories['total_size_gb']:.2f} GB\n\n")
        
        f.write("## Files by Type\n\n")
        for model_type, count in sorted(categories['by_type'].items(), key=lambda x: -x[1]):
            f.write(f"- **{model_type}**: {count}\n")
        f.write("\n")
        
        f.write(f"## Training Checkpoints: {len(categories['checkpoints'])}\n\n")
        for path in categories['checkpoints'][:30]:
            f.write(f"- `{path}`\n")
        f.write("\n")
        
        f.write(f"## Large Models (>100MB): {len(categories['large_models'])}\n\n")
        for item in sorted(categories['large_models'], key=lambda x: -x['size_mb'])[:20]:
            f.write(f"- `{item['path']}` ({item['size_mb']:.2f} MB)\n")
        f.write("\n")
        
        f.write(f"## Very Large Models (>1GB): {len(categories['very_large_models'])}\n\n")
        for item in sorted(categories['very_large_models'], key=lambda x: -x['size_gb'])[:20]:
            f.write(f"- `{item['path']}` ({item['size_gb']:.2f} GB)\n")
        f.write("\n")
        
        f.write(f"## Models with Config Files: {len(categories['with_configs'])}\n\n")
        for item in categories['with_configs'][:20]:
            f.write(f"- `{item['path']}`\n")
            for config in item['configs']:
                f.write(f"  - Config: `{config}`\n")
    
    print(f"Saved Markdown report to: {md_path}")
    print("\n" + "=" * 60)
    print(f"Model file discovery complete!")
    print(f"Found {len(model_files)} model files ({categories['total_size_gb']:.2f} GB)")

if __name__ == '__main__':
    main()
