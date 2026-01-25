#!/usr/bin/env python3
"""
Generate comprehensive master inventory and summary reports.
Combines all discovery data into a single searchable database.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


def load_all_data(discovery_dir: Path) -> Dict:
    """Load all discovery and analysis data."""
    data = {}

    files = {
        "directory_map": "directory_map.json",
        "code_files": "code_files_inventory.json",
        "documentation": "documentation_inventory.json",
        "models": "model_files_inventory.json",
        "value_assessment": "value_assessment.json",
    }

    for key, filename in files.items():
        file_path = discovery_dir / filename
        if file_path.exists():
            with open(file_path) as f:
                data[key] = json.load(f)

    return data


def create_master_inventory(data: Dict) -> Dict:
    """Create master inventory combining all discovery data."""
    inventory = {
        "inventory_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files": 0,
            "code_files": 0,
            "documentation_files": 0,
            "model_files": 0,
            "total_size_gb": 0,
        },
        "files": [],
    }

    # Process code files
    if "code_files" in data:
        code_data = data["code_files"]
        code_files = code_data.get("files", [])
        inventory["summary"]["code_files"] = len(code_files)

        # Get value assessments if available
        value_map = {}
        if "value_assessment" in data:
            for result in data["value_assessment"].get("code_files", []):
                value_map[result["file"]] = result

        for file_info in code_files:
            entry = {
                "type": "code",
                "path": file_info["path"],
                "language": file_info.get("language", "unknown"),
                "size": file_info["size"],
                "size_kb": file_info.get("size_kb", 0),
                "last_modified": file_info.get("last_modified"),
                "value_assessment": value_map.get(file_info["path"], {}).get("assessment", {}),
                "integration_opportunity": value_map.get(file_info["path"], {}).get(
                    "opportunities", {}
                ),
            }
            inventory["files"].append(entry)
            inventory["summary"]["total_size_gb"] += file_info["size"] / (1024**3)

    # Process documentation files
    if "documentation" in data:
        doc_data = data["documentation"]
        doc_files = doc_data.get("files", [])
        inventory["summary"]["documentation_files"] = len(doc_files)

        value_map = {}
        if "value_assessment" in data:
            for result in data["value_assessment"].get("documentation_files", []):
                value_map[result["file"]] = result

        for file_info in doc_files:
            entry = {
                "type": "documentation",
                "path": file_info["path"],
                "doc_type": file_info.get("type", "unknown"),
                "size": file_info["size"],
                "size_kb": file_info.get("size_kb", 0),
                "last_modified": file_info.get("last_modified"),
                "word_count": file_info.get("word_count", 0),
                "value_assessment": value_map.get(file_info["path"], {}).get("assessment", {}),
                "integration_opportunity": value_map.get(file_info["path"], {}).get(
                    "opportunities", {}
                ),
            }
            inventory["files"].append(entry)
            inventory["summary"]["total_size_gb"] += file_info["size"] / (1024**3)

    # Process model files
    if "models" in data:
        model_data = data["models"]
        model_files = model_data.get("files", [])
        inventory["summary"]["model_files"] = len(model_files)

        value_map = {}
        if "value_assessment" in data:
            for result in data["value_assessment"].get("model_files", []):
                value_map[result["file"]] = result

        for file_info in model_files:
            entry = {
                "type": "model",
                "path": file_info["path"],
                "model_type": file_info.get("type", "unknown"),
                "size": file_info["size"],
                "size_mb": file_info.get("size_mb", 0),
                "size_gb": file_info.get("size_gb", 0),
                "last_modified": file_info.get("last_modified"),
                "is_checkpoint": file_info.get("is_checkpoint", False),
                "value_assessment": value_map.get(file_info["path"], {}).get("assessment", {}),
                "integration_opportunity": value_map.get(file_info["path"], {}).get(
                    "opportunities", {}
                ),
            }
            inventory["files"].append(entry)
            inventory["summary"]["total_size_gb"] += file_info["size"] / (1024**3)

    inventory["summary"]["total_files"] = len(inventory["files"])
    inventory["summary"]["total_size_gb"] = round(inventory["summary"]["total_size_gb"], 2)

    return inventory


def generate_summary_report(data: Dict, inventory: Dict) -> str:
    """Generate comprehensive summary report."""
    report = []
    report.append("# Comprehensive File Discovery Summary\n\n")
    report.append(f"**Report Generated**: {datetime.now().isoformat()}\n\n")

    # Overall summary
    report.append("## Overall Summary\n\n")
    report.append(f"- **Total Files Discovered**: {inventory['summary']['total_files']:,}\n")
    report.append(f"- **Code Files**: {inventory['summary']['code_files']:,}\n")
    report.append(f"- **Documentation Files**: {inventory['summary']['documentation_files']:,}\n")
    report.append(f"- **Model Files**: {inventory['summary']['model_files']:,}\n")
    report.append(f"- **Total Size**: {inventory['summary']['total_size_gb']:.2f} GB\n\n")

    # Directory summary
    if "directory_map" in data:
        dir_data = data["directory_map"]
        report.append("## Directory Summary\n\n")
        report.append(f"- **Directories Scanned**: {len(dir_data.get('directories', []))}\n")
        report.append(
            f"- **Total Files in Directories**: {dir_data.get('directories', [{}])[0].get('files', 0) if dir_data.get('directories') else 0:,}\n\n"
        )

    # Code file breakdown
    if "code_files" in data:
        code_data = data["code_files"]
        report.append("## Code Files Breakdown\n\n")
        categories = code_data.get("categories", {})
        by_lang = categories.get("by_language", {})
        report.append("### By Language\n\n")
        for lang, count in sorted(by_lang.items(), key=lambda x: -x[1])[:10]:
            report.append(f"- **{lang}**: {count:,}\n")
        report.append("\n")

        report.append(f"- **KmiDi-Related Files**: {len(categories.get('kmidi_related', []))}\n")
        report.append(f"- **ML-Related Files**: {len(categories.get('ml_related', []))}\n")
        report.append(f"- **Music-Related Files**: {len(categories.get('music_related', []))}\n\n")

    # Value assessment summary
    if "value_assessment" in data:
        value_data = data["value_assessment"]
        report.append("## Value Assessment Summary\n\n")
        by_level = value_data.get("summary", {}).get("by_value_level", {})
        report.append("### Files by Value Level\n\n")
        for level in ["Critical", "High", "Medium", "Low"]:
            count = by_level.get(level, 0)
            report.append(f"- **{level}**: {count}\n")
        report.append("\n")

        integration_ops = value_data.get("summary", {}).get("integration_opportunities", {})
        report.append("### Integration Opportunities\n\n")
        for priority in ["Critical", "High", "Medium"]:
            count = integration_ops.get(priority, 0)
            if count > 0:
                report.append(f"- **{priority} Priority**: {count} files\n")
        report.append("\n")

    # Top recommendations
    report.append("## Top Integration Recommendations\n\n")

    # Get top files by value score
    high_value_files = []
    for entry in inventory["files"]:
        assessment = entry.get("value_assessment", {})
        if assessment.get("value_level") in ["Critical", "High"]:
            opp = entry.get("integration_opportunity", {})
            if opp.get("can_integrate"):
                high_value_files.append(
                    {
                        "path": entry["path"],
                        "type": entry["type"],
                        "value_level": assessment.get("value_level", "Unknown"),
                        "value_score": assessment.get("value_score", 0),
                        "priority": opp.get("integration_priority", "Unknown"),
                    }
                )

    high_value_files.sort(key=lambda x: x["value_score"], reverse=True)

    report.append("### Immediate Integration Opportunities (Critical/High Value)\n\n")
    for i, file_info in enumerate(high_value_files[:30], 1):
        report.append(f"{i}. **{file_info['path']}**\n")
        report.append(f"   - Type: {file_info['type']}\n")
        report.append(
            f"   - Value: {file_info['value_level']} (Score: {file_info['value_score']})\n"
        )
        report.append(f"   - Priority: {file_info['priority']}\n\n")

    return "".join(report)


def main():
    """Main reporting function."""
    print("Generating comprehensive reports...")
    print("=" * 60)

    discovery_dir = Path(__file__).parent.parent.parent / "docs" / "DISCOVERY"
    data = load_all_data(discovery_dir)

    if not data:
        print("No discovery data found. Run discovery scripts first.")
        return

    # Create master inventory
    print("Creating master inventory...")
    inventory = create_master_inventory(data)

    # Save master inventory JSON
    json_path = discovery_dir / "discovery_database.json"
    with open(json_path, "w") as f:
        json.dump(inventory, f, indent=2)
    print(f"Saved master inventory to: {json_path}")

    # Generate master inventory markdown
    md_path = discovery_dir / "MASTER_INVENTORY.md"
    with open(md_path, "w") as f:
        f.write("# Master File Inventory\n\n")
        f.write(f"**Generated**: {inventory['inventory_timestamp']}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total Files**: {inventory['summary']['total_files']:,}\n")
        f.write(f"- **Code Files**: {inventory['summary']['code_files']:,}\n")
        f.write(f"- **Documentation Files**: {inventory['summary']['documentation_files']:,}\n")
        f.write(f"- **Model Files**: {inventory['summary']['model_files']:,}\n")
        f.write(f"- **Total Size**: {inventory['summary']['total_size_gb']:.2f} GB\n\n")

        f.write("## File Inventory\n\n")
        f.write("(See discovery_database.json for complete structured data)\n\n")

        # Group by type
        by_type = {}
        for entry in inventory["files"]:
            file_type = entry["type"]
            if file_type not in by_type:
                by_type[file_type] = []
            by_type[file_type].append(entry)

        for file_type, files in by_type.items():
            f.write(f"### {file_type.title()} Files ({len(files)})\n\n")
            # Show top 50 by value score
            sorted_files = sorted(
                [f for f in files if f.get("value_assessment", {}).get("value_score", 0) > 0],
                key=lambda x: x.get("value_assessment", {}).get("value_score", 0),
                reverse=True,
            )[:50]

            for entry in sorted_files:
                f.write(f"- `{entry['path']}`\n")
                assessment = entry.get("value_assessment", {})
                if assessment:
                    f.write(
                        f"  - Value: {assessment.get('value_level', 'Unknown')} (Score: {assessment.get('value_score', 0)})\n"
                    )

    print(f"Saved master inventory to: {md_path}")

    # Generate summary report
    print("Generating summary report...")
    summary_report = generate_summary_report(data, inventory)
    summary_path = discovery_dir / "DISCOVERY_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write(summary_report)
    print(f"Saved summary report to: {summary_path}")

    print("\n" + "=" * 60)
    print("Report generation complete!")
    print(f"Master inventory: {len(inventory['files'])} files cataloged")


if __name__ == "__main__":
    main()
