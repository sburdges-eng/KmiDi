#!/usr/bin/env python3
"""
KmiDi Environment Loader
========================

Utility module for loading environment variables from the project root.
Loads .env files in priority order:
1. .env (base configuration)
2. Feature-specific files (env/.env.*)
3. .env.local (user overrides, highest priority)

Usage:
    from kmidi_env import load_kmidi_env
    load_kmidi_env()
    
    import os
    api_key = os.getenv('OPENAI_API_KEY')
"""

import os
from pathlib import Path
from typing import Optional, List

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv is required. Install with: pip install python-dotenv"
    )


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root by looking for .env.example or .git directory.
    
    Args:
        start_path: Starting directory (defaults to current file's directory)
        
    Returns:
        Path to project root
    """
    if start_path is None:
        # Default to directory containing this file, then go up to find project root
        start_path = Path(__file__).resolve().parent
    
    current = Path(start_path).resolve()
    
    # Look for project root markers
    markers = ['.env.example', '.git', 'CMakeLists.txt', 'pyproject.toml']
    
    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent
    
    # Fallback: assume we're in project root if .env.example exists in start_path's ancestors
    # or return the start_path's parent (KmiDi-1)
    for parent in Path(start_path).resolve().parents:
        if (parent / '.env.example').exists():
            return parent
    
    # Last resort: return the directory containing this file's parent (assuming we're in python/)
    return Path(__file__).resolve().parent.parent


def load_kmidi_env(
    project_root: Optional[Path] = None,
    features: Optional[List[str]] = None,
    verbose: bool = False
) -> None:
    """
    Load KmiDi environment variables from project root.
    
    Args:
        project_root: Path to project root (auto-detected if None)
        features: List of features to load ('tauri', 'ml', 'training', 'mcp')
        verbose: Print loaded files if True
    """
    if project_root is None:
        project_root = find_project_root()
    
    if features is None:
        features = ['tauri', 'ml', 'training', 'mcp']
    
    # Priority order: base .env, feature files, .env.local
    env_files = []
    
    # 1. Base .env
    base_env = project_root / '.env'
    if base_env.exists():
        env_files.append(base_env)
    
    # 2. Feature-specific files
    feature_map = {
        'tauri': 'env/.env.tauri',
        'ml': 'env/.env.ml',
        'training': 'env/.env.training',
        'mcp': 'env/.env.mcp',
    }
    
    for feature in features:
        if feature in feature_map:
            feature_file = project_root / feature_map[feature]
            if feature_file.exists():
                env_files.append(feature_file)
    
    # 3. .env.local (highest priority)
    local_env = project_root / '.env.local'
    if local_env.exists():
        env_files.append(local_env)
    
    # Load files in order (later files override earlier ones)
    for env_file in env_files:
        if verbose:
            print(f"Loading environment from: {env_file}")
        load_dotenv(env_file, override=False)
    
    # Override with .env.local if it exists
    if local_env.exists():
        load_dotenv(local_env, override=True)


# Auto-load on import (optional, can be disabled)
_AUTO_LOAD = os.getenv('KMIDI_AUTO_LOAD_ENV', 'true').lower() == 'true'

if _AUTO_LOAD:
    try:
        load_kmidi_env(verbose=False)
    except Exception:
        # Silently fail if project root not found or files don't exist
        pass
