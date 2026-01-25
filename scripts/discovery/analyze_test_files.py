#!/usr/bin/env python3
"""
Analyze test files separately - 100,000+ test files found.
Identify valuable test suites, test coverage patterns, and integration opportunities.
"""
import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def analyze_test_files():
    disco_dir = Path("docs/DISCOVERY")
    
    print("=" * 80)
    print("Test Files Analysis")
    print("=" * 80)
    
    # Load code inventory
    with open(disco_dir / "code_files_inventory.json") as f:
        code_inv = json.load(f)
    
    # Find test files
    test_files = defaultdict(list)
    test_stats = defaultdict(int)
    
    for lang, files in code_inv.items():
        for f in files:
            path = f.get('path', '').lower()
            name = f.get('name', '').lower()
            rel_path = f.get('rel_path', '')
            
            # Identify test files
            is_test = False
            test_type = None
            
            if 'test' in path or 'test' in name:
                is_test = True
                if 'test_' in name or name.startswith('test'):
                    test_type = 'pytest'
                elif 'spec' in name:
                    test_type = 'spec'
                elif 'test' in path:
                    test_type = 'test_dir'
            elif 'spec' in name:
                is_test = True
                test_type = 'spec'
            
            if is_test:
                test_files[lang].append({
                    **f,
                    'test_type': test_type,
                    'is_kmidi_related': any(k in rel_path.lower() for k in ['kmidi', 'music', 'kelly', 'penta'])
                })
                test_stats[lang] += 1
                test_stats['total'] += 1
    
    print(f"\nFound {test_stats['total']:,} test files")
    print("\nBy language:")
    for lang in sorted(test_stats.keys()):
        if lang != 'total':
            print(f"  {lang}: {test_stats[lang]:,}")
    
    # Analyze test files
    kmidi_tests = []
    large_tests = []
    recent_tests = []
    
    for lang, files in test_files.items():
        for f in files:
            if f.get('is_kmidi_related'):
                kmidi_tests.append(f)
            if f.get('size', 0) > 50 * 1024:  # > 50KB
                large_tests.append(f)
            # Check recency (last 6 months)
            try:
                from datetime import datetime, timezone, timedelta
                mod_date = datetime.fromisoformat(f.get('modified', ''))
                if mod_date > datetime.now(timezone.utc) - timedelta(days=180):
                    recent_tests.append(f)
            except:
                pass
    
    print(f"\nKmiDi-related tests: {len(kmidi_tests):,}")
    print(f"Large tests (>50KB): {len(large_tests):,}")
    print(f"Recent tests (6 months): {len(recent_tests):,}")
    
    # Save analysis
    output = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'total_tests': test_stats['total'],
            'by_language': dict(test_stats),
            'kmidi_related': len(kmidi_tests),
            'large_tests': len(large_tests),
            'recent_tests': len(recent_tests)
        },
        'kmidi_tests': kmidi_tests[:500],
        'large_tests': sorted(large_tests, key=lambda x: x.get('size', 0), reverse=True)[:200],
        'recent_tests': recent_tests[:500]
    }
    
    output_path = disco_dir / "test_files_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Generate markdown report
    md_path = disco_dir / "test_files_analysis.md"
    with open(md_path, 'w') as f:
        f.write("# Test Files Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total test files:** {test_stats['total']:,}\n")
        f.write(f"- **KmiDi-related:** {len(kmidi_tests):,}\n")
        f.write(f"- **Large tests (>50KB):** {len(large_tests):,}\n")
        f.write(f"- **Recent (6 months):** {len(recent_tests):,}\n\n")
        
        f.write("## Test Files by Language\n\n")
        f.write("| Language | Count |\n")
        f.write("|----------|-------|\n")
        for lang in sorted(test_stats.keys()):
            if lang != 'total':
                f.write(f"| {lang} | {test_stats[lang]:,} |\n")
        
        if kmidi_tests:
            f.write("\n## KmiDi-Related Test Files (Top 100)\n\n")
            f.write("| File | Language | Size (KB) | Path |\n")
            f.write("|------|----------|-----------|------|\n")
            for test in kmidi_tests[:100]:
                size_kb = test.get('size', 0) / 1024
                rel_path = test.get('rel_path', '')
                f.write(f"| `{test.get('name', '')}` | {test.get('language', '')} | {size_kb:.1f} | `{rel_path[:70]}` |\n")
    
    print(f"\n✅ Analysis saved to:")
    print(f"   - {output_path}")
    print(f"   - {md_path}")

if __name__ == "__main__":
    analyze_test_files()
