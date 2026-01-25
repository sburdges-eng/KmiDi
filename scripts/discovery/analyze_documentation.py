#!/usr/bin/env python3
"""
Optimized documentation analysis - focuses on KmiDi-related and high-value docs.
"""
import json
import os
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

KMIDI_KEYWORDS = {'kmidi', 'music', 'emotion', 'kelly', 'companion', 'penta', 'brain', 'training', 'model'}

def analyze_documentation():
    """Analyze documentation files for value."""
    inventory_path = Path("/Users/seanburdges/KmiDi-1/docs/DISCOVERY/documentation_inventory.json")
    
    if not inventory_path.exists():
        print("❌ Documentation inventory not found")
        return
    
    print("Loading documentation inventory...")
    with open(inventory_path) as f:
        doc_data = json.load(f)
    
    # Get all documentation files
    all_docs = []
    for doc_type, files in doc_data.items():
        if isinstance(files, list):
            for file_info in files:
                file_info['doc_type'] = doc_type
                all_docs.append(file_info)
    
    print(f"Found {len(all_docs):,} documentation files")
    
    # Analyze and categorize
    results = {
        'kmidi_related': [],
        'readme_files': [],
        'large_docs': [],
        'recent_docs': [],
        'all_analyzed': []
    }
    
    stats = defaultdict(int)
    cutoff_date = datetime.now().timestamp() - (6 * 30 * 24 * 60 * 60)  # 6 months
    
    print(f"\nAnalyzing documentation files...")
    
    for i, file_info in enumerate(all_docs):
        if (i + 1) % 5000 == 0:
            print(f"  Analyzed {i + 1}/{len(all_docs)} files...")
        
        path_str = file_info['path'].lower()
        
        # Check if KmiDi-related
        is_kmidi = any(keyword in path_str for keyword in KMIDI_KEYWORDS)
        
        # Check if README
        is_readme = 'readme' in file_info['name'].lower()
        
        # Check size
        size_mb = file_info.get('size_mb', 0)
        is_large = size_mb > 1.0
        
        # Check modification date
        try:
            mod_date = datetime.fromisoformat(file_info.get('modified', ''))
            is_recent = mod_date.timestamp() > cutoff_date
        except:
            is_recent = False
        
        analysis = {
            **file_info,
            'kmidi_related': is_kmidi,
            'is_readme': is_readme,
            'is_large': is_large,
            'is_recent': is_recent
        }
        
        results['all_analyzed'].append(analysis)
        stats['total'] += 1
        
        if is_kmidi:
            results['kmidi_related'].append(analysis)
            stats['kmidi_related'] += 1
        
        if is_readme:
            results['readme_files'].append(analysis)
            stats['readme'] += 1
        
        if is_large:
            results['large_docs'].append(analysis)
            stats['large'] += 1
        
        if is_recent:
            results['recent_docs'].append(analysis)
            stats['recent'] += 1
    
    # Sort results
    results['kmidi_related'].sort(key=lambda x: x.get('size_mb', 0), reverse=True)
    results['readme_files'].sort(key=lambda x: x.get('size_mb', 0), reverse=True)
    results['large_docs'].sort(key=lambda x: x.get('size_mb', 0), reverse=True)
    results['recent_docs'].sort(key=lambda x: x.get('modified', ''), reverse=True)
    
    results['summary'] = dict(stats)
    results['summary']['generated'] = datetime.now().isoformat()
    
    # Save results
    output_dir = Path("/Users/seanburdges/KmiDi-1/docs/DISCOVERY")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open(output_dir / "documentation_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save Markdown report
    with open(output_dir / "documentation_analysis.md", "w") as f:
        f.write("# Documentation Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Analyzed:** {stats['total']:,}\n")
        f.write(f"- **KmiDi-Related:** {stats['kmidi_related']:,}\n")
        f.write(f"- **README Files:** {stats['readme']:,}\n")
        f.write(f"- **Large Docs (>1MB):** {stats['large']:,}\n")
        f.write(f"- **Recent (6 months):** {stats['recent']:,}\n\n")
        
        if results['kmidi_related']:
            f.write("## KmiDi-Related Documentation (Top 100)\n\n")
            f.write("| File | Type | Size (MB) | Modified | Path |\n")
            f.write("|------|------|-----------|----------|------|\n")
            for item in results['kmidi_related'][:100]:
                rel_path = item['path'].replace('/Users/seanburdges/', '')
                f.write(f"| `{item['name']}` | {item.get('doc_type', 'unknown')} | "
                       f"{item.get('size_mb', 0):.2f} | {item.get('modified', '')[:10]} | "
                       f"`{rel_path[:60]}...` |\n")
            f.write("\n")
        
        if results['readme_files']:
            f.write("## README Files (Top 50)\n\n")
            f.write("| File | Size (MB) | Modified | Path |\n")
            f.write("|------|-----------|----------|------|\n")
            for item in results['readme_files'][:50]:
                rel_path = item['path'].replace('/Users/seanburdges/', '')
                f.write(f"| `{item['name']}` | {item.get('size_mb', 0):.2f} | "
                       f"{item.get('modified', '')[:10]} | `{rel_path[:60]}...` |\n")
            f.write("\n")
    
    print(f"\n✅ Documentation analysis complete")
    print(f"   - Total: {stats['total']:,}")
    print(f"   - KmiDi-related: {stats['kmidi_related']:,}")
    print(f"   - README files: {stats['readme']:,}")
    print(f"✅ Results saved to {output_dir / 'documentation_analysis.md'}")

if __name__ == "__main__":
    analyze_documentation()
