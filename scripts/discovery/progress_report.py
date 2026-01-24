#!/usr/bin/env python3
"""Generate a progress report for the comprehensive discovery process."""

import json
import os
from datetime import datetime
from pathlib import Path

def format_size(size_bytes):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def main():
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "docs" / "DISCOVERY"

    print("=" * 70)
    print("COMPREHENSIVE FILE DISCOVERY PROGRESS REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    tasks = [
        ("Task 1.1", "Directory structure discovery", out_dir / "directory_map.md"),
        ("Task 1.2", "Project-like directories", out_dir / "project_directories.md"),
        ("Task 1.3", "Large file/directories report", out_dir / "large_directories.md"),
        ("Task 2.1", "Code files inventory", out_dir / "code_files_inventory.md"),
        ("Task 2.2", "Valuable code files", out_dir / "valuable_code_files.md"),
        ("Task 2.3", "KmiDi-related code", out_dir / "kmidi_related_code.md"),
        ("Task 3.1", "Documentation inventory", out_dir / "documentation_inventory.md"),
        ("Task 3.2", "Valuable documentation", out_dir / "valuable_documentation.md"),
        ("Task 3.3", "Documentation insights", out_dir / "documentation_insights.md"),
        ("Task 4.1", "Model files inventory", out_dir / "model_files_inventory.md"),
        ("Task 4.2", "Valuable models", out_dir / "valuable_models.md"),
        ("Task 4.3", "Training artifacts", out_dir / "training_artifacts.md"),
        ("Task 5.1", "Content analysis", out_dir / "content_analysis.md"),
        ("Task 5.2", "Relationship mapping", out_dir / "file_relationships.md"),
        ("Task 5.3", "Duplicates & variants", out_dir / "duplicates_and_variants.md"),
        ("Task 6.1", "Value categorization", out_dir / "value_categorization.md"),
        ("Task 6.2", "Integration opportunities", out_dir / "integration_opportunities.md"),
        ("Task 6.3", "Integration recommendations", out_dir / "integration_recommendations.md"),
        ("Task 7.1", "Master inventory", out_dir / "MASTER_INVENTORY.md"),
        ("Task 7.2", "Summary report", out_dir / "DISCOVERY_SUMMARY.md"),
        ("Task 7.3", "Searchable database", out_dir / "discovery_database.json"),
    ]

    completed = []
    pending = []

    report_lines = [
        "# Discovery Progress Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "| Task | Status | Output |",
        "|------|--------|--------|",
    ]

    for task_id, name, path in tasks:
        if path.exists():
            completed.append(task_id)
            report_lines.append(f"| {task_id} | ✅ | `{path.relative_to(repo_root)}` |")
        else:
            pending.append(task_id)
            report_lines.append(f"| {task_id} | ⏳ | `{path.relative_to(repo_root)}` |")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Completed: {len(completed)}/{len(tasks)} tasks")
    print(f"⏳ Pending: {len(pending)}/{len(tasks)} tasks")
    
    completion_pct = (len(completed) / len(tasks)) * 100 if tasks else 0
    print(f"📊 Progress: {completion_pct:.1f}%")
    
    report_lines += [
        "",
        f"**Completed:** {len(completed)}/{len(tasks)} ({completion_pct:.1f}%)",
        "",
    ]

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    if pending:
        print("Re-run:")
        print("  python3 scripts/discovery/generate_reports.py")
        print("  python3 scripts/discovery/analyze_files.py")
        print("  python3 scripts/discovery/postprocess_reports.py")
    else:
        print("All outputs present.")
    print("=" * 70)

    out_path = out_dir / "PROGRESS_REPORT.md"
    out_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    print(f"\n✅ Wrote {out_path}")

if __name__ == '__main__':
    main()
