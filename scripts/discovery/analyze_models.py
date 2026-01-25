#!/usr/bin/env python3
"""
Analyze discovered ML model files for value assessment.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def analyze_models():
    """Analyze model files for value indicators."""
    inventory_path = Path("/Users/seanburdges/KmiDi-1/docs/DISCOVERY/model_files_inventory.json")
    
    if not inventory_path.exists():
        print("❌ Model inventory not found. Run find_models.py first.")
        return
    
    with open(inventory_path) as f:
        models_by_type = json.load(f)
    
    # Analysis categories
    kmidi_related = []
    large_models = []  # > 100MB
    recent_models = []  # Modified in last 6 months
    training_artifacts = []
    config_files = defaultdict(list)
    
    cutoff_date = datetime.now().timestamp() - (6 * 30 * 24 * 60 * 60)  # 6 months ago
    
    for model_type, files in models_by_type.items():
        for file_info in files:
            path = file_info['path']
            size_mb = file_info['size_mb']
            modified_ts = datetime.fromisoformat(file_info['modified']).timestamp()
            
            # Check if KmiDi-related
            if 'kmidi' in path.lower() or 'music' in path.lower() or 'emotion' in path.lower():
                file_info['value_indicator'] = 'kmidi_related'
                kmidi_related.append(file_info)
            
            # Large models
            if size_mb > 100:
                file_info['value_indicator'] = 'large_model'
                large_models.append(file_info)
            
            # Recent models
            if modified_ts > cutoff_date:
                file_info['value_indicator'] = 'recent'
                recent_models.append(file_info)
            
            # Training artifacts (check for associated files)
            model_dir = Path(path).parent
            config_patterns = ['config.json', 'config.yaml', 'hyperparameters.json', 'train_config.json']
            for config_file in config_patterns:
                config_path = model_dir / config_file
                if config_path.exists():
                    file_info['has_config'] = True
                    config_files[path].append(str(config_path))
                    training_artifacts.append(file_info)
                    break
    
    # Generate analysis report
    output_dir = Path("/Users/seanburdges/KmiDi-1/docs/DISCOVERY")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis = {
        'summary': {
            'total_models': sum(len(files) for files in models_by_type.values()),
            'kmidi_related': len(kmidi_related),
            'large_models': len(large_models),
            'recent_models': len(recent_models),
            'training_artifacts': len(training_artifacts),
            'generated': datetime.now().isoformat()
        },
        'kmidi_related': kmidi_related[:100],  # Top 100
        'large_models': sorted(large_models, key=lambda x: x['size_mb'], reverse=True)[:100],
        'recent_models': sorted(recent_models, key=lambda x: x['modified'], reverse=True)[:100],
        'training_artifacts': training_artifacts[:100],
        'config_files': dict(list(config_files.items())[:50])
    }
    
    # Save JSON
    with open(output_dir / "model_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Save Markdown report
    with open(output_dir / "model_analysis.md", "w") as f:
        f.write("# ML Model Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Models:** {analysis['summary']['total_models']:,}\n")
        f.write(f"- **KmiDi-Related:** {analysis['summary']['kmidi_related']}\n")
        f.write(f"- **Large Models (>100MB):** {analysis['summary']['large_models']}\n")
        f.write(f"- **Recent Models (6 months):** {analysis['summary']['recent_models']}\n")
        f.write(f"- **Training Artifacts:** {analysis['summary']['training_artifacts']}\n\n")
        
        if kmidi_related:
            f.write("## KmiDi-Related Models\n\n")
            f.write("| File | Size (MB) | Modified | Path |\n")
            f.write("|------|-----------|----------|------|\n")
            for model in kmidi_related[:50]:
                rel_path = model['path'].replace('/Users/seanburdges/', '')
                f.write(f"| `{model['name']}` | {model['size_mb']:.1f} | "
                       f"{model['modified'][:10]} | `{rel_path[:60]}...` |\n")
            f.write("\n")
        
        if large_models:
            f.write("## Large Models (>100MB)\n\n")
            f.write("| File | Size (MB) | Modified | Path |\n")
            f.write("|------|-----------|----------|------|\n")
            for model in sorted(large_models, key=lambda x: x['size_mb'], reverse=True)[:50]:
                rel_path = model['path'].replace('/Users/seanburdges/', '')
                f.write(f"| `{model['name']}` | {model['size_mb']:.1f} | "
                       f"{model['modified'][:10]} | `{rel_path[:60]}...` |\n")
            f.write("\n")
    
    print(f"✅ Model analysis complete")
    print(f"   - KmiDi-related: {len(kmidi_related)}")
    print(f"   - Large models: {len(large_models)}")
    print(f"   - Recent models: {len(recent_models)}")
    print(f"✅ Results saved to {output_dir / 'model_analysis.md'}")

if __name__ == "__main__":
    analyze_models()
