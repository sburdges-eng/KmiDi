#!/usr/bin/env python3
"""
Find and catalog all documentation files across the MacBook Pro.
Analyzes content for value, structure, and insights.
"""

import argparse
import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

# Documentation extensions
DOC_EXTENSIONS = {
    'markdown': {'.md', '.markdown'},
    'text': {'.txt', '.rst'},
    'html': {'.html', '.htm'},
    'pdf': {'.pdf'},
    'other': {'.org', '.wiki', '.adoc'},
}

ALL_DOC_EXTENSIONS = set()
for exts in DOC_EXTENSIONS.values():
    ALL_DOC_EXTENSIONS.update(exts)

# Keywords to identify valuable documentation
VALUE_KEYWORDS = {'TODO', 'FIXME', 'IMPORTANT', 'NOTE', 'WARNING', 'DEPRECATED'}
STRUCTURE_KEYWORDS = {'##', '###', 'Table of Contents', 'Overview', 'Introduction', 'API', 'Architecture'}

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

def get_doc_type(file_path: Path) -> str:
    """Determine documentation type from extension."""
    ext = file_path.suffix.lower()
    for doc_type, exts in DOC_EXTENSIONS.items():
        if ext in exts:
            return doc_type
    return 'unknown'

def analyze_markdown_file(file_path: Path) -> Dict:
    """Analyze Markdown file for structure and content."""
    analysis = {
        'has_toc': False,
        'heading_count': 0,
        'code_block_count': 0,
        'link_count': 0,
        'image_count': 0,
        'value_keywords': [],
        'word_count': 0,
        'line_count': 0,
    }
    
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')
        analysis['line_count'] = len(lines)
        
        # Check for table of contents
        content_lower = content.lower()
        analysis['has_toc'] = 'table of contents' in content_lower or '## contents' in content_lower
        
        # Count headings
        analysis['heading_count'] = len(re.findall(r'^#+\s+', content, re.MULTILINE))
        
        # Count code blocks
        analysis['code_block_count'] = content.count('```')
        
        # Count links
        analysis['link_count'] = len(re.findall(r'\[([^\]]+)\]\([^\)]+\)', content))
        
        # Count images
        analysis['image_count'] = len(re.findall(r'!\[([^\]]*)\]\([^\)]+\)', content))
        
        # Find value keywords
        for keyword in VALUE_KEYWORDS:
            if keyword in content:
                analysis['value_keywords'].append(keyword)
        
        # Word count (approximate)
        words = re.findall(r'\b\w+\b', content)
        analysis['word_count'] = len(words)
        
    except Exception as e:
        pass
    
    return analysis

def analyze_text_file(file_path: Path) -> Dict:
    """Analyze text file for content."""
    analysis = {
        'line_count': 0,
        'word_count': 0,
        'value_keywords': [],
    }
    
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')
        analysis['line_count'] = len(lines)
        
        # Find value keywords
        for keyword in VALUE_KEYWORDS:
            if keyword in content:
                analysis['value_keywords'].append(keyword)
        
        # Word count
        words = re.findall(r'\b\w+\b', content)
        analysis['word_count'] = len(words)
        
    except Exception as e:
        pass
    
    return analysis

def analyze_file(file_path: Path) -> Dict:
    """Analyze a documentation file."""
    try:
        stat = file_path.stat()
        doc_type = get_doc_type(file_path)
        
        file_info = {
            'path': str(file_path),
            'name': file_path.name,
            'type': doc_type,
            'size': stat.st_size,
            'size_kb': round(stat.st_size / 1024, 2),
            'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'directory': str(file_path.parent),
        }
        
        # Type-specific analysis
        if doc_type == 'markdown':
            analysis = analyze_markdown_file(file_path)
            file_info.update(analysis)
        elif doc_type == 'text':
            analysis = analyze_text_file(file_path)
            file_info.update(analysis)
        
        return file_info
    
    except Exception as e:
        return None

def find_documentation_files(root: Path = Path('/Users/seanburdges'), max_files: int | None = None) -> List[Dict]:
    """Find all documentation files. max_files=None means no limit."""
    doc_files = []
    
    print(f"Scanning for documentation files starting from: {root}" + (f" (max {max_files})" if max_files else " (no limit)"))
    
    for file_path in root.rglob('*'):
        if max_files is not None and len(doc_files) >= max_files:
            print(f"Reached limit of {max_files} files")
            break
        
        if should_exclude(file_path):
            continue
        
        if file_path.is_file() and file_path.suffix.lower() in ALL_DOC_EXTENSIONS:
            file_info = analyze_file(file_path)
            if file_info:
                doc_files.append(file_info)
                
                if len(doc_files) % 500 == 0 and len(doc_files) > 0:
                    print(f"Found {len(doc_files)} documentation files...")
    
    return doc_files

def categorize_files(doc_files: List[Dict]) -> Dict:
    """Categorize documentation files."""
    categories = {
        'by_type': defaultdict(int),
        'with_toc': [],
        'large_files': [],  # > 100KB
        'recent_files': [],
        'with_value_keywords': [],
        'comprehensive': [],  # > 1000 words
    }
    
    cutoff_date = datetime.now().timestamp() - (30 * 24 * 60 * 60)
    
    for file_info in doc_files:
        categories['by_type'][file_info['type']] += 1
        
        if file_info.get('has_toc'):
            categories['with_toc'].append(file_info['path'])
        
        if file_info['size'] > 100 * 1024:
            categories['large_files'].append({
                'path': file_info['path'],
                'size_kb': file_info['size_kb']
            })
        
        try:
            mod_time = datetime.fromisoformat(file_info['last_modified']).timestamp()
            if mod_time > cutoff_date:
                categories['recent_files'].append(file_info['path'])
        except:
            pass
        
        if file_info.get('value_keywords'):
            categories['with_value_keywords'].append({
                'path': file_info['path'],
                'keywords': file_info['value_keywords']
            })
        
        if file_info.get('word_count', 0) > 1000:
            categories['comprehensive'].append({
                'path': file_info['path'],
                'word_count': file_info['word_count']
            })
    
    return categories

def main():
    """Main execution function."""
    print("Finding all documentation files...")
    print("=" * 60)
    parser = argparse.ArgumentParser(description="Find and catalog documentation files")
    parser.add_argument("--max-files", type=int, default=None, help="Cap number of files (default: no limit)")
    parser.add_argument("--root", type=Path, default=Path("/Users/seanburdges"), help="Root directory to scan")
    args = parser.parse_args()

    doc_files = find_documentation_files(root=args.root, max_files=args.max_files)
    categories = categorize_files(doc_files)
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'docs' / 'DISCOVERY'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / 'documentation_inventory.json'
    with open(json_path, 'w') as f:
        json.dump({
            'scan_timestamp': datetime.now().isoformat(),
            'total_files': len(doc_files),
            'files': doc_files,
            'categories': categories
        }, f, indent=2)
    print(f"\nSaved JSON results to: {json_path}")
    
    # Save Markdown report
    md_path = output_dir / 'documentation_inventory.md'
    with open(md_path, 'w') as f:
        f.write("# Documentation Inventory\n\n")
        f.write(f"**Scan Date**: {datetime.now().isoformat()}\n\n")
        f.write(f"**Total Documentation Files Found**: {len(doc_files):,}\n\n")
        
        f.write("## Files by Type\n\n")
        for doc_type, count in sorted(categories['by_type'].items(), key=lambda x: -x[1]):
            f.write(f"- **{doc_type}**: {count:,}\n")
        f.write("\n")
        
        f.write(f"## Files with Table of Contents: {len(categories['with_toc'])}\n\n")
        for path in categories['with_toc'][:30]:
            f.write(f"- `{path}`\n")
        f.write("\n")
        
        f.write(f"## Comprehensive Documents (>1000 words): {len(categories['comprehensive'])}\n\n")
        for item in sorted(categories['comprehensive'], key=lambda x: -x['word_count'])[:30]:
            f.write(f"- `{item['path']}` ({item['word_count']:,} words)\n")
        f.write("\n")
        
        f.write(f"## Files with Value Keywords: {len(categories['with_value_keywords'])}\n\n")
        for item in categories['with_value_keywords'][:30]:
            keywords = ', '.join(item['keywords'])
            f.write(f"- `{item['path']}` ({keywords})\n")
    
    print(f"Saved Markdown report to: {md_path}")
    print("\n" + "=" * 60)
    print(f"Documentation discovery complete!")
    print(f"Found {len(doc_files):,} documentation files")

if __name__ == '__main__':
    main()
