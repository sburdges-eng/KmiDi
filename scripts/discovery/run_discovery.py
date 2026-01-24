#!/usr/bin/env python3
"""
Master script to run all discovery tasks.
Orchestrates the comprehensive file discovery and analysis process.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_script(script_path: Path, description: str):
    """Run a discovery script and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"\n✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        print(f"⏱️  {description} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    """Run all discovery scripts in sequence."""
    scripts_dir = Path(__file__).parent
    output_dir = scripts_dir.parent.parent / 'docs' / 'DISCOVERY'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("COMPREHENSIVE MACBOOK PRO FILE DISCOVERY")
    print("="*60)
    print(f"Start Time: {datetime.now().isoformat()}")
    print(f"Output Directory: {output_dir}")
    
    scripts = [
        (scripts_dir / 'scan_directories.py', 'Directory Structure Discovery'),
        (scripts_dir / 'find_code_files.py', 'Code File Discovery'),
        (scripts_dir / 'find_documentation.py', 'Documentation Discovery'),
        (scripts_dir / 'find_models.py', 'ML Model Discovery'),
    ]
    
    results = {}
    for script_path, description in scripts:
        if script_path.exists():
            success = run_script(script_path, description)
            results[description] = success
        else:
            print(f"⚠️  Script not found: {script_path}")
            results[description] = False
    
    # Summary
    print("\n" + "="*60)
    print("DISCOVERY SUMMARY")
    print("="*60)
    
    for description, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {description}")
    
    all_success = all(results.values())
    print(f"\nOverall Status: {'✅ ALL TASKS COMPLETED' if all_success else '⚠️  SOME TASKS FAILED'}")
    print(f"End Time: {datetime.now().isoformat()}")
    
    return 0 if all_success else 1

if __name__ == '__main__':
    sys.exit(main())
