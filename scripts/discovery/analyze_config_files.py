#!/usr/bin/env python3
"""
Analyze configuration files separately.
Identify valuable configs, build systems, and integration opportunities.
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def analyze_config_files():
    disco_dir = Path("docs/DISCOVERY")
    
    print("=" * 80)
    print("Configuration Files Analysis")
    print("=" * 80)
    
    # Load inventories
    with open(disco_dir / "code_files_inventory.json") as f:
        code_inv = json.load(f)
    
    with open(disco_dir / "discovery_database.json") as f:
        db = json.load(f)
    
    # Config file extensions
    config_extensions = {
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'cfg',
        '.conf': 'conf',
        '.config': 'config',
        '.properties': 'properties',
        '.env': 'env'
    }
    
    # Config file names (even without extension)
    config_names = {
        'cmakelists.txt': 'cmake',
        'makefile': 'make',
        'dockerfile': 'docker',
        '.gitignore': 'git',
        '.gitattributes': 'git',
        'package.json': 'node',
        'package-lock.json': 'node',
        'requirements.txt': 'python',
        'pyproject.toml': 'python',
        'setup.py': 'python',
        'pom.xml': 'java',
        'build.gradle': 'java',
        'cargo.toml': 'rust',
        'go.mod': 'go',
        'composer.json': 'php',
        '.clang-format': 'clang',
        '.clang-tidy': 'clang',
        '.editorconfig': 'editor',
        '.prettierrc': 'prettier',
        'tsconfig.json': 'typescript',
        'webpack.config.js': 'webpack',
        'babel.config.js': 'babel'
    }
    
    config_files = defaultdict(list)
    config_stats = defaultdict(int)
    
    # Find config files in code inventory
    for lang, files in code_inv.items():
        for f in files:
            path = Path(f.get('path', ''))
            name = path.name.lower()
            ext = path.suffix.lower()
            
            is_config = False
            config_type = None
            
            # Check by extension
            if ext in config_extensions:
                is_config = True
                config_type = config_extensions[ext]
            
            # Check by name
            if name in config_names:
                is_config = True
                config_type = config_names[name]
            
            if is_config:
                rel_path = f.get('rel_path', '')
                config_files[config_type].append({
                    **f,
                    'config_type': config_type,
                    'is_kmidi_related': any(k in rel_path.lower() for k in ['kmidi', 'music', 'kelly', 'penta', 'brain'])
                })
                config_stats[config_type] += 1
                config_stats['total'] += 1
    
    print(f"\nFound {config_stats['total']:,} configuration files")
    print("\nBy type:")
    for config_type in sorted(config_stats.keys()):
        if config_type != 'total':
            print(f"  {config_type}: {config_stats[config_type]:,}")
    
    # Analyze configs
    kmidi_configs = []
    build_configs = []
    project_configs = []
    
    build_types = ['cmake', 'make', 'docker', 'node', 'python', 'java', 'rust', 'go']
    project_types = ['git', 'editor', 'prettier', 'clang']
    
    for config_type, files in config_files.items():
        for f in files:
            if f.get('is_kmidi_related'):
                kmidi_configs.append(f)
            if config_type in build_types:
                build_configs.append(f)
            if config_type in project_types:
                project_configs.append(f)
    
    print(f"\nKmiDi-related configs: {len(kmidi_configs):,}")
    print(f"Build system configs: {len(build_configs):,}")
    print(f"Project configs: {len(project_configs):,}")
    
    # Save analysis
    output = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'total_configs': config_stats['total'],
            'by_type': dict(config_stats),
            'kmidi_related': len(kmidi_configs),
            'build_configs': len(build_configs),
            'project_configs': len(project_configs)
        },
        'kmidi_configs': kmidi_configs[:200],
        'build_configs': build_configs[:200],
        'project_configs': project_configs[:200]
    }
    
    output_path = disco_dir / "config_files_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Generate markdown report
    md_path = disco_dir / "config_files_analysis.md"
    with open(md_path, 'w') as f:
        f.write("# Configuration Files Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total config files:** {config_stats['total']:,}\n")
        f.write(f"- **KmiDi-related:** {len(kmidi_configs):,}\n")
        f.write(f"- **Build system configs:** {len(build_configs):,}\n")
        f.write(f"- **Project configs:** {len(project_configs):,}\n\n")
        
        f.write("## Config Files by Type\n\n")
        f.write("| Type | Count |\n")
        f.write("|------|-------|\n")
        for config_type in sorted(config_stats.keys()):
            if config_type != 'total':
                f.write(f"| {config_type} | {config_stats[config_type]:,} |\n")
        
        if kmidi_configs:
            f.write("\n## KmiDi-Related Config Files (Top 50)\n\n")
            f.write("| File | Type | Size (KB) | Path |\n")
            f.write("|------|------|-----------|------|\n")
            for config in kmidi_configs[:50]:
                size_kb = config.get('size', 0) / 1024
                rel_path = config.get('rel_path', '')
                f.write(f"| `{config.get('name', '')}` | {config.get('config_type', '')} | {size_kb:.1f} | `{rel_path[:70]}` |\n")
    
    print(f"\n✅ Analysis saved to:")
    print(f"   - {output_path}")
    print(f"   - {md_path}")

if __name__ == "__main__":
    analyze_config_files()
