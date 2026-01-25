#!/usr/bin/env python3
"""
Create comprehensive analyses for remaining gaps:
1. Integration readiness assessment (extend to code/docs)
2. File recency analysis
3. Directory-level value scoring
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

def parse_iso(ts: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)

def assess_readiness(file_info, file_type='code'):
    """Assess integration readiness for a file."""
    path = file_info.get('path', '').lower()
    rel_path = file_info.get('rel_path', '').lower()
    
    # Check if in vendor/system directories
    vendor_patterns = ['site-packages', 'node_modules', '.venv', 'venv', '.cache', '.gradle']
    if any(p in path for p in vendor_patterns):
        return 'excluded'
    
    # Check if already in KmiDi-1
    if rel_path.startswith('kmidi-1/'):
        return 'already_integrated'
    
    # Check for KmiDi-related keywords
    kmidi_keywords = ['kmidi', 'music_brain', 'penta_core', 'kelly', 'emotion']
    is_kmidi = any(k in rel_path for k in kmidi_keywords)
    
    # Readiness scoring
    readiness_score = 0
    
    if is_kmidi:
        readiness_score += 3
    
    # Check file type
    if file_type == 'model':
        # Models are generally copy-ready
        readiness_score += 2
    elif file_type == 'code':
        # Code may need adaptation
        readiness_score += 1
    
    # Size consideration (smaller files easier to integrate)
    size_mb = file_info.get('size', 0) / (1024 * 1024)
    if size_mb < 1:
        readiness_score += 1
    elif size_mb > 100:
        readiness_score -= 1
    
    # Determine readiness level
    if readiness_score >= 4:
        return 'ready'
    elif readiness_score >= 2:
        return 'needs_review'
    elif readiness_score >= 0:
        return 'needs_adaptation'
    else:
        return 'low_priority'

def categorize_by_age(modified_iso: str) -> str:
    """Categorize file by age."""
    try:
        mod_date = parse_iso(modified_iso)
        now = datetime.now(timezone.utc)
        age = now - mod_date
        
        if age < timedelta(days=7):
            return 'last_week'
        elif age < timedelta(days=30):
            return 'last_month'
        elif age < timedelta(days=180):
            return 'last_6_months'
        elif age < timedelta(days=365):
            return 'last_year'
        elif age < timedelta(days=730):
            return '1_2_years'
        else:
            return 'older_than_2_years'
    except:
        return 'unknown'

def main():
    disco_dir = Path("docs/DISCOVERY")
    
    print("=" * 80)
    print("Comprehensive Gap Analysis")
    print("=" * 80)
    
    # Load data
    with open(disco_dir / "discovery_database.json") as f:
        db = json.load(f)
    
    with open(disco_dir / "code_analysis.json") as f:
        code_analysis = json.load(f)
    
    with open(disco_dir / "documentation_analysis.json") as f:
        doc_analysis = json.load(f)
    
    with open(disco_dir / "model_analysis.json") as f:
        model_analysis = json.load(f)
    
    # 1. Integration Readiness Assessment
    print("\n1. Assessing integration readiness...")
    
    readiness_results = {
        'code': defaultdict(list),
        'documentation': defaultdict(list),
        'models': defaultdict(list)
    }
    
    # Code files
    for item in code_analysis.get('all_analyzed', [])[:1000]:
        readiness = assess_readiness(item, 'code')
        readiness_results['code'][readiness].append(item)
    
    # Documentation
    for item in doc_analysis.get('all_analyzed', [])[:1000]:
        readiness = assess_readiness(item, 'documentation')
        readiness_results['documentation'][readiness].append(item)
    
    # Models
    for item in model_analysis.get('kmidi_related', [])[:500]:
        readiness = assess_readiness(item, 'model')
        readiness_results['models'][readiness].append(item)
    
    print(f"   Code: {sum(len(v) for v in readiness_results['code'].values())} files assessed")
    print(f"   Documentation: {sum(len(v) for v in readiness_results['documentation'].values())} files assessed")
    print(f"   Models: {sum(len(v) for v in readiness_results['models'].values())} files assessed")
    
    # 2. File Recency Analysis
    print("\n2. Analyzing file recency...")
    
    recency_by_type = {
        'code': defaultdict(int),
        'documentation': defaultdict(int),
        'models': defaultdict(int)
    }
    
    # Code
    for item in code_analysis.get('all_analyzed', []):
        age_bucket = categorize_by_age(item.get('modified', ''))
        recency_by_type['code'][age_bucket] += 1
    
    # Documentation
    for item in doc_analysis.get('all_analyzed', []):
        age_bucket = categorize_by_age(item.get('modified', ''))
        recency_by_type['documentation'][age_bucket] += 1
    
    # Models
    for item in model_analysis.get('kmidi_related', []):
        age_bucket = categorize_by_age(item.get('modified', ''))
        recency_by_type['models'][age_bucket] += 1
    
    print("   Recency buckets populated")
    
    # 3. Directory-Level Value Scoring
    print("\n3. Calculating directory-level value scores...")
    
    # Aggregate file scores by directory
    dir_scores = defaultdict(lambda: {
        'total_files': 0,
        'total_score': 0.0,
        'high_value_files': 0,
        'kmidi_files': 0,
        'categories': defaultdict(int)
    })
    
    # Code files
    for item in code_analysis.get('all_analyzed', []):
        rel_path = item.get('rel_path', '')
        if not rel_path:
            continue
        dir_path = str(Path(rel_path).parent)
        score = item.get('value_score', 0) or item.get('complexity_score', 0) or 0
        
        dir_scores[dir_path]['total_files'] += 1
        dir_scores[dir_path]['total_score'] += score
        if score > 50:
            dir_scores[dir_path]['high_value_files'] += 1
        if item.get('kmidi_related'):
            dir_scores[dir_path]['kmidi_files'] += 1
    
    # Documentation
    for item in doc_analysis.get('all_analyzed', []):
        rel_path = item.get('rel_path', '')
        if not rel_path:
            continue
        dir_path = str(Path(rel_path).parent)
        score = item.get('value_score', 0) or 0
        
        dir_scores[dir_path]['total_files'] += 1
        dir_scores[dir_path]['total_score'] += score
        if score > 50:
            dir_scores[dir_path]['high_value_files'] += 1
        if item.get('kmidi_related'):
            dir_scores[dir_path]['kmidi_files'] += 1
    
    # Calculate average scores
    top_directories = []
    for dir_path, stats in dir_scores.items():
        if stats['total_files'] > 0:
            avg_score = stats['total_score'] / stats['total_files']
            top_directories.append({
                'path': dir_path,
                'avg_score': round(avg_score, 2),
                'total_score': round(stats['total_score'], 2),
                'total_files': stats['total_files'],
                'high_value_files': stats['high_value_files'],
                'kmidi_files': stats['kmidi_files']
            })
    
    top_directories.sort(key=lambda x: x['avg_score'], reverse=True)
    
    print(f"   Analyzed {len(top_directories)} directories")
    
    # Save results
    output = {
        'generated_at': datetime.now().isoformat(),
        'integration_readiness': {
            'code': {k: len(v) for k, v in readiness_results['code'].items()},
            'documentation': {k: len(v) for k, v in readiness_results['documentation'].items()},
            'models': {k: len(v) for k, v in readiness_results['models'].items()}
        },
        'recency_analysis': recency_by_type,
        'directory_scores': top_directories[:200]
    }
    
    output_path = disco_dir / "comprehensive_gap_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Generate markdown reports
    md_path = disco_dir / "comprehensive_gap_analysis.md"
    with open(md_path, 'w') as f:
        f.write("# Comprehensive Gap Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        # Integration Readiness
        f.write("## Integration Readiness Assessment\n\n")
        f.write("### Code Files\n\n")
        f.write("| Readiness | Count |\n")
        f.write("|-----------|-------|\n")
        for readiness in ['ready', 'needs_review', 'needs_adaptation', 'low_priority', 'excluded', 'already_integrated']:
            count = len(readiness_results['code'].get(readiness, []))
            f.write(f"| {readiness} | {count} |\n")
        
        f.write("\n### Documentation\n\n")
        f.write("| Readiness | Count |\n")
        f.write("|-----------|-------|\n")
        for readiness in ['ready', 'needs_review', 'needs_adaptation', 'low_priority', 'excluded', 'already_integrated']:
            count = len(readiness_results['documentation'].get(readiness, []))
            f.write(f"| {readiness} | {count} |\n")
        
        # Recency Analysis
        f.write("\n## File Recency Analysis\n\n")
        f.write("### Code Files\n\n")
        f.write("| Age Bucket | Count |\n")
        f.write("|------------|-------|\n")
        for bucket in ['last_week', 'last_month', 'last_6_months', 'last_year', '1_2_years', 'older_than_2_years']:
            count = recency_by_type['code'].get(bucket, 0)
            f.write(f"| {bucket.replace('_', ' ').title()} | {count:,} |\n")
        
        # Directory Scores
        f.write("\n## Top Valuable Directories\n\n")
        f.write("| Directory | Avg Score | Total Files | High Value | KmiDi Files |\n")
        f.write("|-----------|-----------|-------------|------------|-------------|\n")
        for dir_info in top_directories[:50]:
            f.write(f"| `{dir_info['path'][:60]}` | {dir_info['avg_score']} | {dir_info['total_files']} | "
                   f"{dir_info['high_value_files']} | {dir_info['kmidi_files']} |\n")
    
    print(f"\n✅ Comprehensive analysis saved to:")
    print(f"   - {output_path}")
    print(f"   - {md_path}")

if __name__ == "__main__":
    main()
