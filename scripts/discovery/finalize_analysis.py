#!/usr/bin/env python3
"""
Comprehensive analysis and integration planning.
Combines: content analysis, relationship mapping, duplicate detection,
value categorization, and integration recommendations.
"""
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_analysis_results():
    """Load all analysis results."""
    base_dir = Path("/Users/seanburdges/KmiDi-1/docs/DISCOVERY")
    
    results = {}
    
    # Load code analysis
    code_path = base_dir / "code_analysis.json"
    if code_path.exists():
        with open(code_path) as f:
            results['code'] = json.load(f)
    
    # Load documentation analysis
    doc_path = base_dir / "documentation_analysis.json"
    if doc_path.exists():
        with open(doc_path) as f:
            results['documentation'] = json.load(f)
    
    # Load model analysis
    model_path = base_dir / "model_analysis.json"
    if model_path.exists():
        with open(model_path) as f:
            results['models'] = json.load(f)
    
    return results

def categorize_by_value(analysis_results):
    """Categorize findings by value (Critical, High, Medium, Low)."""
    categories = {
        'critical': [],
        'high': [],
        'medium': [],
        'low': []
    }
    
    # Code files
    if 'code' in analysis_results:
        code_data = analysis_results['code']
        for item in code_data.get('kmidi_related', []):
            score = 0
            if item.get('kmidi_related'):
                score += 10
            if item.get('complexity_score', 0) > 50:
                score += 5
            if item.get('todos'):
                score += 2
            
            if score >= 15:
                categories['critical'].append(('code', item))
            elif score >= 10:
                categories['high'].append(('code', item))
            elif score >= 5:
                categories['medium'].append(('code', item))
            else:
                categories['low'].append(('code', item))
    
    # Documentation
    if 'documentation' in analysis_results:
        doc_data = analysis_results['documentation']
        for item in doc_data.get('kmidi_related', [])[:1000]:  # Top 1000
            score = 0
            if item.get('kmidi_related'):
                score += 8
            if item.get('is_readme'):
                score += 5
            if item.get('size_mb', 0) > 0.1:
                score += 2
            
            if score >= 10:
                categories['high'].append(('documentation', item))
            elif score >= 5:
                categories['medium'].append(('documentation', item))
            else:
                categories['low'].append(('documentation', item))
    
    # Models
    if 'models' in analysis_results:
        model_data = analysis_results['models']
        for item in model_data.get('kmidi_related', [])[:500]:  # Top 500
            score = 0
            if item.get('size_mb', 0) > 100:
                score += 10
            elif item.get('size_mb', 0) > 10:
                score += 5
            
            if score >= 10:
                categories['critical'].append(('model', item))
            elif score >= 5:
                categories['high'].append(('model', item))
            else:
                categories['medium'].append(('model', item))
    
    return categories

def identify_integration_opportunities(categories):
    """Identify integration opportunities for KmiDi-1."""
    opportunities = {
        'models': [],
        'code_modules': [],
        'documentation': [],
        'configs': []
    }
    
    # Critical and high value items are integration opportunities
    for item_type, item in categories['critical'] + categories['high'][:200]:
        path = item.get('path', '')
        
        if 'model' in path.lower() or item_type == 'model':
            opportunities['models'].append({
                'path': path,
                'type': item_type,
                'value': 'critical' if (item_type, item) in categories['critical'] else 'high',
                'reason': 'High-value model file'
            })
        elif item_type == 'code':
            # Check if it's a module/package
            if '__init__.py' in path or Path(path).suffix == '.py':
                opportunities['code_modules'].append({
                    'path': path,
                    'type': item_type,
                    'value': 'critical' if (item_type, item) in categories['critical'] else 'high',
                    'reason': f"Complexity: {item.get('complexity_score', 0)}, Classes: {len(item.get('classes', []))}"
                })
        elif item_type == 'documentation':
            opportunities['documentation'].append({
                'path': path,
                'type': item_type,
                'value': 'high',
                'reason': 'KmiDi-related documentation'
            })
    
    return opportunities

def generate_integration_recommendations(opportunities):
    """Generate prioritized integration recommendations."""
    recommendations = []
    
    # Models - highest priority
    for model in opportunities['models'][:20]:
        recommendations.append({
            'priority': 1,
            'category': 'ML Models',
            'action': f"Review and potentially integrate model: {Path(model['path']).name}",
            'path': model['path'],
            'reason': model['reason'],
            'effort': 'Medium',
            'value': model['value']
        })
    
    # Code modules
    for code in opportunities['code_modules'][:30]:
        recommendations.append({
            'priority': 2,
            'category': 'Code Modules',
            'action': f"Evaluate code module for integration: {Path(code['path']).name}",
            'path': code['path'],
            'reason': code['reason'],
            'effort': 'High',
            'value': code['value']
        })
    
    # Documentation
    for doc in opportunities['documentation'][:20]:
        recommendations.append({
            'priority': 3,
            'category': 'Documentation',
            'action': f"Review documentation: {Path(doc['path']).name}",
            'path': doc['path'],
            'reason': doc['reason'],
            'effort': 'Low',
            'value': doc['value']
        })
    
    # Sort by priority
    recommendations.sort(key=lambda x: x['priority'])
    
    return recommendations

def create_master_inventory(analysis_results, categories, opportunities, recommendations):
    """Create master inventory of all discovered files."""
    master = {
        'summary': {
            'generated': datetime.now().isoformat(),
            'total_code_files': len(analysis_results.get('code', {}).get('all_analyzed', [])),
            'total_doc_files': len(analysis_results.get('documentation', {}).get('all_analyzed', [])),
            'total_model_files': analysis_results.get('models', {}).get('summary', {}).get('total_models', 0),
            'critical_items': len(categories['critical']),
            'high_value_items': len(categories['high']),
            'integration_opportunities': sum(len(v) for v in opportunities.values()),
            'recommendations': len(recommendations)
        },
        'categories': {
            'critical': len(categories['critical']),
            'high': len(categories['high']),
            'medium': len(categories['medium']),
            'low': len(categories['low'])
        },
        'integration_opportunities': opportunities,
        'recommendations': recommendations
    }
    
    return master

def main():
    """Main function."""
    print("=" * 80)
    print("Comprehensive Analysis & Integration Planning")
    print("=" * 80)
    
    # Load analysis results
    print("\n1. Loading analysis results...")
    analysis_results = load_analysis_results()
    print(f"   ✓ Loaded {len(analysis_results)} analysis files")
    
    # Categorize by value
    print("\n2. Categorizing by value...")
    categories = categorize_by_value(analysis_results)
    print(f"   ✓ Critical: {len(categories['critical'])}")
    print(f"   ✓ High: {len(categories['high'])}")
    print(f"   ✓ Medium: {len(categories['medium'])}")
    print(f"   ✓ Low: {len(categories['low'])}")
    
    # Identify integration opportunities
    print("\n3. Identifying integration opportunities...")
    opportunities = identify_integration_opportunities(categories)
    total_opps = sum(len(v) for v in opportunities.values())
    print(f"   ✓ Found {total_opps} integration opportunities")
    print(f"     - Models: {len(opportunities['models'])}")
    print(f"     - Code modules: {len(opportunities['code_modules'])}")
    print(f"     - Documentation: {len(opportunities['documentation'])}")
    
    # Generate recommendations
    print("\n4. Generating integration recommendations...")
    recommendations = generate_integration_recommendations(opportunities)
    print(f"   ✓ Generated {len(recommendations)} recommendations")
    
    # Create master inventory
    print("\n5. Creating master inventory...")
    master = create_master_inventory(analysis_results, categories, opportunities, recommendations)
    
    # Save results
    output_dir = Path("/Users/seanburdges/KmiDi-1/docs/DISCOVERY")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open(output_dir / "master_inventory.json", "w") as f:
        json.dump(master, f, indent=2)
    
    # Save CSV for searchable database
    import csv
    with open(output_dir / "integration_recommendations.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['priority', 'category', 'action', 'path', 'reason', 'effort', 'value'])
        writer.writeheader()
        writer.writerows(recommendations)
    
    # Generate summary report
    print("\n6. Generating summary report...")
    with open(output_dir / "INTEGRATION_SUMMARY.md", "w") as f:
        f.write("# KmiDi-1 Integration Summary Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"This report summarizes the comprehensive file discovery and analysis ")
        f.write(f"conducted across your MacBook Pro to identify valuable assets for ")
        f.write(f"integration into the KmiDi-1 project.\n\n")
        
        f.write("### Key Findings\n\n")
        f.write(f"- **Code Files Analyzed:** {master['summary']['total_code_files']:,}\n")
        f.write(f"- **Documentation Files:** {master['summary']['total_doc_files']:,}\n")
        f.write(f"- **ML Model Files:** {master['summary']['total_model_files']:,}\n")
        f.write(f"- **Critical Items:** {master['summary']['critical_items']}\n")
        f.write(f"- **High Value Items:** {master['summary']['high_value_items']}\n")
        f.write(f"- **Integration Opportunities:** {master['summary']['integration_opportunities']}\n")
        f.write(f"- **Recommendations:** {master['summary']['recommendations']}\n\n")
        
        f.write("## Value Categories\n\n")
        f.write(f"- **Critical:** {master['categories']['critical']} items\n")
        f.write(f"- **High:** {master['categories']['high']} items\n")
        f.write(f"- **Medium:** {master['categories']['medium']} items\n")
        f.write(f"- **Low:** {master['categories']['low']} items\n\n")
        
        f.write("## Top Integration Recommendations\n\n")
        f.write("| Priority | Category | Action | Effort | Value |\n")
        f.write("|----------|----------|--------|--------|-------|\n")
        for rec in recommendations[:30]:
            f.write(f"| {rec['priority']} | {rec['category']} | {rec['action'][:50]}... | "
                   f"{rec['effort']} | {rec['value']} |\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Review the integration recommendations in `integration_recommendations.csv`\n")
        f.write("2. Prioritize based on project needs and available resources\n")
        f.write("3. Begin integration with highest priority items\n")
        f.write("4. Update this report as integration progresses\n\n")
    
    print(f"\n✅ Analysis complete!")
    print(f"   - Master inventory: {output_dir / 'master_inventory.json'}")
    print(f"   - Recommendations CSV: {output_dir / 'integration_recommendations.csv'}")
    print(f"   - Summary report: {output_dir / 'INTEGRATION_SUMMARY.md'}")

if __name__ == "__main__":
    main()
