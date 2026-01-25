#!/usr/bin/env python3
"""
Optimized code file analysis - focuses on high-value files.
Prioritizes: Python files, KmiDi-related, recent files, excludes system dirs.
"""
import json
import os
import ast
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# System directories to skip
SKIP_DIRS = {
    '.git', '.venv', 'venv', '__pycache__', 'node_modules',
    '.pytest_cache', '.mypy_cache', '.tox', '.nox',
    'build', 'dist', 'target', '.gradle', '.idea', '.vscode',
    'Library', 'System', 'Applications', '.Trash',
    '.cache', '.local', '.config', '.cursor', '.cargo', '.rustup',
    '.npm', '.claude', 'OneDrive', 'Google Drive'
}

# KmiDi-related keywords
KMIDI_KEYWORDS = {'kmidi', 'music', 'emotion', 'kelly', 'companion', 'penta', 'brain'}

def should_analyze_file(file_path):
    """Check if file should be analyzed."""
    path_str = str(file_path)
    parts = Path(file_path).parts
    
    # Skip system directories
    if any(skip in parts for skip in SKIP_DIRS):
        return False
    
    # Prioritize Python files
    if file_path.suffix != '.py':
        return False
    
    return True

def analyze_python_file(file_path):
    """Analyze a Python file for value indicators."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return None
    
    if not content.strip():
        return None
    
    analysis = {
        'path': str(file_path),
        'name': file_path.name,
        'size': file_path.stat().st_size,
        'lines': len(content.splitlines()),
        'imports': [],
        'classes': [],
        'functions': [],
        'todos': [],
        'kmidi_related': False,
        'complexity_score': 0
    }
    
    # Check for KmiDi-related keywords
    content_lower = content.lower()
    if any(keyword in content_lower for keyword in KMIDI_KEYWORDS):
        analysis['kmidi_related'] = True
    
    # Parse AST for structure
    try:
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            # Collect imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis['imports'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis['imports'].append(node.module)
            
            # Collect classes
            if isinstance(node, ast.ClassDef):
                analysis['classes'].append(node.name)
                analysis['complexity_score'] += 2
            
            # Collect functions
            if isinstance(node, ast.FunctionDef):
                analysis['functions'].append(node.name)
                analysis['complexity_score'] += 1
        
        # Count complexity indicators
        analysis['complexity_score'] += len(analysis['classes']) * 2
        analysis['complexity_score'] += len(analysis['functions'])
        
    except SyntaxError:
        pass
    
    # Find TODOs/FIXMEs
    for i, line in enumerate(content.splitlines(), 1):
        if re.search(r'\bTODO\b|\bFIXME\b|\bXXX\b|\bHACK\b', line, re.IGNORECASE):
            analysis['todos'].append({'line': i, 'text': line.strip()[:100]})
    
    return analysis

def sample_files_from_inventory():
    """Sample high-value files from the code inventory."""
    inventory_path = Path("/Users/seanburdges/KmiDi-1/docs/DISCOVERY/code_files_inventory.json")
    
    if not inventory_path.exists():
        print("❌ Code inventory not found")
        return []
    
    print("Loading code inventory...")
    with open(inventory_path) as f:
        code_data = json.load(f)
    
    # Get Python files
    python_files = code_data.get('python', [])
    print(f"Found {len(python_files):,} Python files in inventory")
    
    # Filter and prioritize
    priority_files = []
    regular_files = []
    
    for file_info in python_files:
        file_path = Path(file_info['path'])
        
        if not should_analyze_file(file_path):
            continue
        
        # Check if KmiDi-related
        path_str = str(file_path).lower()
        is_kmidi = any(keyword in path_str for keyword in KMIDI_KEYWORDS)
        
        file_info['file_path'] = file_path
        file_info['is_kmidi_related'] = is_kmidi
        
        if is_kmidi:
            priority_files.append(file_info)
        else:
            regular_files.append(file_info)
    
    # Sample strategy:
    # - All KmiDi-related files (up to 5000)
    # - Top 5000 by size/modification from others
    # - Focus on recent files (last 6 months)
    
    cutoff_date = datetime.now().timestamp() - (6 * 30 * 24 * 60 * 60)
    
    # Sort regular files by modification date and size
    regular_files.sort(key=lambda x: (
        datetime.fromisoformat(x['modified']).timestamp() > cutoff_date,
        x.get('size', 0)
    ), reverse=True)
    
    # Combine: all priority + top regular
    selected = priority_files[:5000] + regular_files[:5000]
    
    print(f"Selected {len(selected)} files for analysis:")
    print(f"  - KmiDi-related: {len(priority_files[:5000])}")
    print(f"  - Other high-value: {len(regular_files[:5000])}")
    
    return selected

def analyze_code_files():
    """Analyze selected code files."""
    selected_files = sample_files_from_inventory()
    
    if not selected_files:
        print("No files selected for analysis")
        return
    
    print(f"\nAnalyzing {len(selected_files)} files...")
    
    results = {
        'kmidi_related': [],
        'high_complexity': [],
        'with_todos': [],
        'recent': [],
        'all_analyzed': []
    }
    
    stats = defaultdict(int)
    
    for i, file_info in enumerate(selected_files):
        if (i + 1) % 100 == 0:
            print(f"  Analyzed {i + 1}/{len(selected_files)} files...")
        
        file_path = file_info.get('file_path') or Path(file_info['path'])
        
        if not file_path.exists():
            continue
        
        analysis = analyze_python_file(file_path)
        if not analysis:
            continue
        
        # Add metadata from inventory
        analysis['modified'] = file_info.get('modified', '')
        analysis['size_kb'] = file_info.get('size', 0) / 1024
        
        results['all_analyzed'].append(analysis)
        stats['total_analyzed'] += 1
        
        # Categorize
        if analysis['kmidi_related']:
            results['kmidi_related'].append(analysis)
            stats['kmidi_related'] += 1
        
        if analysis['complexity_score'] > 20:
            results['high_complexity'].append(analysis)
            stats['high_complexity'] += 1
        
        if analysis['todos']:
            results['with_todos'].append(analysis)
            stats['with_todos'] += 1
        
        # Recent (last 6 months)
        try:
            mod_date = datetime.fromisoformat(analysis['modified'])
            cutoff = datetime.now() - timedelta(days=180)
            if mod_date > cutoff:
                results['recent'].append(analysis)
                stats['recent'] += 1
        except:
            pass
    
    # Save results
    output_dir = Path("/Users/seanburdges/KmiDi-1/docs/DISCOVERY")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort results
    for key in results:
        if key != 'all_analyzed':
            results[key].sort(key=lambda x: x.get('complexity_score', 0), reverse=True)
    
    results['summary'] = dict(stats)
    results['summary']['generated'] = datetime.now().isoformat()
    
    # Save JSON
    with open(output_dir / "code_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save Markdown report
    with open(output_dir / "code_analysis.md", "w") as f:
        f.write("# Code File Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Analyzed:** {stats['total_analyzed']:,}\n")
        f.write(f"- **KmiDi-Related:** {stats['kmidi_related']:,}\n")
        f.write(f"- **High Complexity:** {stats['high_complexity']:,}\n")
        f.write(f"- **With TODOs:** {stats['with_todos']:,}\n")
        f.write(f"- **Recent (6 months):** {stats['recent']:,}\n\n")
        
        if results['kmidi_related']:
            f.write("## KmiDi-Related Files (Top 50)\n\n")
            f.write("| File | Lines | Complexity | Classes | Functions | TODOs | Path |\n")
            f.write("|------|-------|------------|---------|-----------|-------|------|\n")
            for item in results['kmidi_related'][:50]:
                rel_path = item['path'].replace('/Users/seanburdges/', '')
                f.write(f"| `{item['name']}` | {item['lines']} | {item['complexity_score']} | "
                       f"{len(item['classes'])} | {len(item['functions'])} | {len(item['todos'])} | "
                       f"`{rel_path[:50]}...` |\n")
            f.write("\n")
        
        if results['high_complexity']:
            f.write("## High Complexity Files (Top 50)\n\n")
            f.write("| File | Complexity | Lines | Classes | Functions | Path |\n")
            f.write("|------|------------|-------|---------|-----------|------|\n")
            for item in results['high_complexity'][:50]:
                rel_path = item['path'].replace('/Users/seanburdges/', '')
                f.write(f"| `{item['name']}` | {item['complexity_score']} | {item['lines']} | "
                       f"{len(item['classes'])} | {len(item['functions'])} | "
                       f"`{rel_path[:50]}...` |\n")
            f.write("\n")
    
    print(f"\n✅ Analysis complete")
    print(f"   - Analyzed: {stats['total_analyzed']:,} files")
    print(f"   - KmiDi-related: {stats['kmidi_related']:,}")
    print(f"   - High complexity: {stats['high_complexity']:,}")
    print(f"✅ Results saved to {output_dir / 'code_analysis.md'}")

if __name__ == "__main__":
    from datetime import timedelta
    analyze_code_files()
