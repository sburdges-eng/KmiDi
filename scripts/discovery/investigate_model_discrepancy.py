#!/usr/bin/env python3
"""
Investigate and reconcile model count discrepancy.
Compare discovery_database.json with model_files_inventory.json to find missing models.
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def main():
    disco_dir = Path("docs/DISCOVERY")
    db_path = disco_dir / "discovery_database.json"
    inv_path = disco_dir / "model_files_inventory.json"
    
    print("=" * 80)
    print("Model Count Discrepancy Investigation")
    print("=" * 80)
    
    # Load both files
    with open(db_path) as f:
        db = json.load(f)
    with open(inv_path) as f:
        inv = json.load(f)
    
    db_models = db.get('inventories', {}).get('models', {})
    db_total = sum(len(v) for v in db_models.values())
    inv_total = sum(len(v) for v in inv.values())
    
    print(f"\nDatabase models: {db_total:,}")
    print(f"Inventory models: {inv_total:,}")
    print(f"Missing: {db_total - inv_total:,}\n")
    
    # Find missing models
    inv_paths = set()
    for model_list in inv.values():
        for model in model_list:
            inv_paths.add(model.get('path', ''))
    
    missing_by_type = defaultdict(list)
    for model_type, model_list in db_models.items():
        for model in model_list:
            path = model.get('path', '')
            if path not in inv_paths:
                missing_by_type[model_type].append(model)
    
    print("Missing models by type:")
    for model_type in sorted(missing_by_type.keys()):
        count = len(missing_by_type[model_type])
        print(f"  {model_type}: {count:,} files")
    
    # Analyze missing models
    print("\n" + "=" * 80)
    print("Analysis of Missing Models")
    print("=" * 80)
    
    # Check if missing models are in excluded directories
    excluded_patterns = [
        'site-packages', 'node_modules', '.venv', 'venv',
        '.cache', '.gradle', '.cargo', '.rustup', '.npm',
        'build/', 'dist/', 'target/', '__pycache__'
    ]
    
    excluded = defaultdict(int)
    included = []
    
    for model_type, models in missing_by_type.items():
        for model in models:
            path = model.get('path', '').lower()
            is_excluded = any(pattern in path for pattern in excluded_patterns)
            if is_excluded:
                excluded[model_type] += 1
            else:
                included.append((model_type, model))
    
    print(f"\nExcluded (vendor/system): {sum(excluded.values()):,}")
    for model_type, count in sorted(excluded.items()):
        print(f"  {model_type}: {count:,}")
    
    print(f"\nPotentially valuable (not excluded): {len(included):,}")
    
    # Sample of potentially valuable missing models
    if included:
        print("\nSample of potentially valuable missing models (top 20 by size):")
        included_sorted = sorted(included, key=lambda x: x[1].get('size', 0), reverse=True)[:20]
        for model_type, model in included_sorted:
            rel_path = model.get('rel_path', model.get('path', ''))
            size_mb = model.get('size', 0) / (1024 * 1024)
            print(f"  {model_type}: {size_mb:.2f} MB - {rel_path[:80]}")
    
    # Save missing models analysis
    output = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'db_total': db_total,
            'inv_total': inv_total,
            'missing_total': db_total - inv_total,
            'excluded_total': sum(excluded.values()),
            'potentially_valuable': len(included)
        },
        'missing_by_type': {k: len(v) for k, v in missing_by_type.items()},
        'excluded_by_type': dict(excluded),
        'potentially_valuable_samples': [
            {
                'type': model_type,
                'path': model.get('path'),
                'rel_path': model.get('rel_path'),
                'size_mb': round(model.get('size', 0) / (1024 * 1024), 2),
                'modified': model.get('modified')
            }
            for model_type, model in included_sorted[:100]
        ]
    }
    
    output_path = disco_dir / "model_discrepancy_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Generate markdown report
    md_path = disco_dir / "model_discrepancy_analysis.md"
    with open(md_path, 'w') as f:
        f.write("# Model Count Discrepancy Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Database models:** {db_total:,}\n")
        f.write(f"- **Inventory models:** {inv_total:,}\n")
        f.write(f"- **Missing:** {db_total - inv_total:,}\n")
        f.write(f"- **Excluded (vendor/system):** {sum(excluded.values()):,}\n")
        f.write(f"- **Potentially valuable:** {len(included):,}\n\n")
        
        f.write("## Missing Models by Type\n\n")
        f.write("| Type | Missing | Excluded | Potentially Valuable |\n")
        f.write("|------|---------|----------|---------------------|\n")
        for model_type in sorted(set(list(missing_by_type.keys()) + list(excluded.keys()))):
            missing = len(missing_by_type.get(model_type, []))
            excl = excluded.get(model_type, 0)
            valuable = missing - excl
            f.write(f"| {model_type} | {missing:,} | {excl:,} | {valuable:,} |\n")
        
        if included_sorted:
            f.write("\n## Potentially Valuable Missing Models (Top 100)\n\n")
            f.write("| Type | Size (MB) | Modified | Path |\n")
            f.write("|------|-----------|----------|------|\n")
            for model_type, model in included_sorted[:100]:
                rel_path = model.get('rel_path', model.get('path', ''))
                size_mb = model.get('size', 0) / (1024 * 1024)
                modified = model.get('modified', '')[:10]
                f.write(f"| {model_type} | {size_mb:.2f} | {modified} | `{rel_path[:80]}` |\n")
    
    print(f"\n✅ Analysis saved to:")
    print(f"   - {output_path}")
    print(f"   - {md_path}")

if __name__ == "__main__":
    main()
