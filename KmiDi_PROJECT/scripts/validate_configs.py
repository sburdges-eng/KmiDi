#!/usr/bin/env python3
"""
Validate all YAML configuration files.

Tests:
- YAML syntax validity
- Required fields present
- External path handling
- Environment variable substitution
"""

import sys
import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent


def find_yaml_files(config_dir: Path) -> List[Path]:
    """Find all YAML files in config directory."""
    yaml_files = []
    for pattern in ['*.yaml', '*.yml']:
        yaml_files.extend(config_dir.glob(pattern))
    return sorted(yaml_files)


def validate_yaml_syntax(file_path: Path) -> tuple[bool, Optional[str]]:
    """Validate YAML file syntax."""
    try:
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def check_tabs_vs_spaces(file_path: Path) -> tuple[bool, Optional[str]]:
    """Check for tab characters (should use spaces)."""
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if '\t' in line:
                    return False, f"Line {line_num} contains tab character (should use spaces)"
        return True, None
    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_paths(config: dict, file_path: Path) -> List[str]:
    """Validate that paths in config are handled correctly."""
    issues = []
    
    def check_path(value, key_path=""):
        """Recursively check paths in config."""
        if isinstance(value, dict):
            for k, v in value.items():
                check_path(v, f"{key_path}.{k}" if key_path else k)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                check_path(item, f"{key_path}[{i}]")
        elif isinstance(value, str):
            # Check for external SSD paths
            if 'data_path' in key_path.lower() or 'path' in key_path.lower():
                if value.startswith('/Volumes/'):
                    # External SSD path - check if exists or has fallback
                    if '${KELLY_AUDIO_DATA_ROOT' not in value and not Path(value).exists():
                        # Check for environment variable fallback
                        env_var = os.getenv('KELLY_AUDIO_DATA_ROOT')
                        if not env_var:
                            issues.append(
                                f"External path '{value}' at '{key_path}' may not exist. "
                                "Consider using ${KELLY_AUDIO_DATA_ROOT} with fallback."
                            )
    
    check_path(config)
    return issues


def validate_required_fields(config: dict, file_path: Path) -> List[str]:
    """Check for common required fields."""
    issues = []
    file_name = file_path.name
    
    # Check for device/backend configs
    if 'build' in file_name or 'train' in file_name:
        if 'device' not in config and 'backend' not in config:
            issues.append("Build/train config should have 'device' or 'backend' field")
    
    # Check for model configs
    if any(model in file_name for model in ['emotion', 'harmony', 'melody', 'groove', 'dynamics']):
        if 'model' not in config and 'architecture' not in config:
            issues.append(f"Model config '{file_name}' should have 'model' or 'architecture' field")
    
    return issues


def validate_config_file(file_path: Path) -> Dict:
    """Validate a single config file."""
    result = {
        'file': str(file_path.relative_to(project_root)),
        'valid': True,
        'errors': [],
        'warnings': [],
    }
    
    # Check YAML syntax
    syntax_ok, syntax_error = validate_yaml_syntax(file_path)
    if not syntax_ok:
        result['valid'] = False
        result['errors'].append(f"YAML syntax error: {syntax_error}")
        return result
    
    # Check for tabs
    tabs_ok, tabs_error = check_tabs_vs_spaces(file_path)
    if not tabs_ok:
        result['warnings'].append(tabs_error)
    
    # Load and validate content
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            result['warnings'].append("Config file is empty")
            return result
        
        # Validate paths
        path_issues = validate_paths(config, file_path)
        result['warnings'].extend(path_issues)
        
        # Validate required fields
        field_issues = validate_required_fields(config, file_path)
        result['warnings'].extend(field_issues)
        
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Error loading config: {e}")
    
    return result


def main():
    config_dir = project_root / 'config'
    
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        return 1
    
    print("="*70)
    print("YAML Configuration Validation")
    print("="*70)
    print(f"Checking config files in: {config_dir}\n")
    
    yaml_files = find_yaml_files(config_dir)
    
    if not yaml_files:
        print("No YAML files found in config directory")
        return 1
    
    results = []
    for yaml_file in yaml_files:
        result = validate_config_file(yaml_file)
        results.append(result)
    
    # Report results
    valid_count = sum(1 for r in results if r['valid'])
    total_count = len(results)
    
    print(f"Checked {total_count} YAML files\n")
    
    for result in results:
        status = "✓" if result['valid'] and not result['warnings'] else "⚠" if result['valid'] else "✗"
        print(f"{status} {result['file']}")
        
        if result['errors']:
            for error in result['errors']:
                print(f"    ERROR: {error}")
        
        if result['warnings']:
            for warning in result['warnings']:
                print(f"    WARNING: {warning}")
    
    print(f"\n{'='*70}")
    print(f"Results: {valid_count}/{total_count} files valid")
    
    error_count = sum(1 for r in results if r['errors'])
    warning_count = sum(len(r['warnings']) for r in results)
    
    if error_count > 0:
        print(f"Errors: {error_count}")
    if warning_count > 0:
        print(f"Warnings: {warning_count}")
    
    if valid_count == total_count and error_count == 0:
        print("\n✓ All config files are valid!")
        return 0
    else:
        print(f"\n⚠ {total_count - valid_count} files have errors, {warning_count} warnings")
        return 1 if error_count > 0 else 0


if __name__ == '__main__':
    sys.exit(main())

