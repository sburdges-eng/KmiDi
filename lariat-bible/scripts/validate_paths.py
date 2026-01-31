#!/usr/bin/env python3
"""
Simple validator: checks directories listed in DIRECTORY_STRUCTURE.txt
and scans README.md for local path patterns to verify existence.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def check_directories():
    missing = []
    ds_file = ROOT / "DIRECTORY_STRUCTURE.txt"
    if not ds_file.exists():
        print(f"Missing: {ds_file}")
        return [str(ds_file)]
    text = ds_file.read_text()
    for line in text.splitlines():
        m = re.match(r'^\s*[├└]\──\s+([^/]+/)', line)
        if m:
            d = ROOT / m.group(1).strip()
            if not d.exists():
                missing.append(str(d))
    return missing

def scan_readme():
    missing = []
    readme = ROOT / "README.md"
    if not readme.exists():
        return [str(readme)]
    text = readme.read_text()
    # find simple relative paths like data/invoices or 01_VENDOR_MANAGEMENT/
    paths = set(re.findall(r'([A-Za-z0-9_\-./]+/?[A-Za-z0-9_\-./]*)', text))
    # filter likely directories (heuristic)
    candidates = [p for p in paths if '/' in p and len(p) < 200]
    for p in candidates:
        p = p.strip()
        # only check top-level referenced paths
        if p.startswith(("http://", "https://")):
            continue
        path = (ROOT / p).resolve()
        if not path.exists():
            missing.append(str(path))
    return sorted(set(missing))

def main():
    print(f"Validating repo at: {ROOT}")
    misses = []
    misses += check_directories()
    misses += scan_readme()
    if misses:
        print("\nMISSING ITEMS:")
        for m in sorted(set(misses)):
            print(" -", m)
        sys.exit(2)
    print("Validation passed: no missing directories or README-referenced paths found.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
