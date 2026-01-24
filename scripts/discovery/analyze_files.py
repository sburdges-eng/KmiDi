#!/usr/bin/env python3
"""
Analyze discovered files for value assessment and integration opportunities.
Categorizes files by value level and identifies integration opportunities.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict


def load_discovery_data(discovery_dir: Path) -> Dict:
    """Load all discovery JSON files."""
    data = {}

    files = {
        "directory_map": discovery_dir / "directory_map.json",
        "code_files": discovery_dir / "code_files_inventory.json",
        "documentation": discovery_dir / "documentation_inventory.json",
        "models": discovery_dir / "model_files_inventory.json",
    }

    for key, file_path in files.items():
        if file_path.exists():
            with open(file_path) as f:
                data[key] = json.load(f)
        else:
            print(f"Warning: {file_path} not found")

    return data


def assess_code_file_value(file_info: Dict, kmidi_paths: set) -> Dict:
    """Assess the value of a code file."""
    value_score = 0
    value_level = "Low"
    reasons = []

    # Check if in KmiDi-related directories
    path = file_info["path"]
    is_kmidi_related = any(kp in path for kp in kmidi_paths)

    # Size factor (larger files often more valuable)
    if file_info["size_kb"] > 100:
        value_score += 2
        reasons.append("Large file (>100KB)")
    elif file_info["size_kb"] > 10:
        value_score += 1

    # KmiDi-related imports/includes
    if file_info.get("has_kmidi_imports") or file_info.get("has_kmidi_includes"):
        value_score += 5
        reasons.append("KmiDi-related imports")
        value_level = "High"

    # ML keywords
    if file_info.get("has_ml_keywords"):
        value_score += 3
        reasons.append("ML/AI related")

    # Music keywords
    if file_info.get("has_music_keywords"):
        value_score += 3
        reasons.append("Music-related")

    # Complexity indicators
    classes = len(file_info.get("classes", []))
    functions = len(file_info.get("functions", []))
    if classes > 5 or functions > 10:
        value_score += 2
        reasons.append("High complexity")

    # TODOs indicate active development
    if file_info.get("todo_count", 0) > 0:
        value_score += 1
        reasons.append(f"{file_info['todo_count']} TODOs")

    # Recent modification
    try:
        mod_time = datetime.fromisoformat(file_info["last_modified"])
        days_ago = (datetime.now() - mod_time).days
        if days_ago < 30:
            value_score += 2
            reasons.append("Recently modified")
        elif days_ago < 90:
            value_score += 1
    except:
        pass

    # Determine value level
    if value_score >= 8:
        value_level = "Critical"
    elif value_score >= 5:
        value_level = "High"
    elif value_score >= 3:
        value_level = "Medium"

    return {
        "value_score": value_score,
        "value_level": value_level,
        "reasons": reasons,
        "is_kmidi_related": is_kmidi_related,
    }


def assess_documentation_value(file_info: Dict) -> Dict:
    """Assess the value of a documentation file."""
    value_score = 0
    value_level = "Low"
    reasons = []

    # Size factor
    if file_info.get("word_count", 0) > 1000:
        value_score += 3
        reasons.append("Comprehensive (>1000 words)")
    elif file_info.get("word_count", 0) > 500:
        value_score += 2
    elif file_info.get("word_count", 0) > 100:
        value_score += 1

    # Structure indicators
    if file_info.get("has_toc"):
        value_score += 2
        reasons.append("Has table of contents")

    if file_info.get("heading_count", 0) > 10:
        value_score += 2
        reasons.append("Well-structured")

    # Value keywords
    if file_info.get("value_keywords"):
        value_score += 2
        reasons.append(f"Contains: {', '.join(file_info['value_keywords'])}")

    # Code blocks indicate technical docs
    if file_info.get("code_block_count", 0) > 0:
        value_score += 1
        reasons.append("Contains code examples")

    # Determine value level
    if value_score >= 6:
        value_level = "Critical"
    elif value_score >= 4:
        value_level = "High"
    elif value_score >= 2:
        value_level = "Medium"

    return {"value_score": value_score, "value_level": value_level, "reasons": reasons}


def assess_model_value(file_info: Dict) -> Dict:
    """Assess the value of a model file."""
    value_score = 0
    value_level = "Low"
    reasons = []

    # Size factor (larger models often more valuable)
    if file_info["size_gb"] > 1:
        value_score += 5
        reasons.append("Very large model (>1GB)")
        value_level = "High"
    elif file_info["size_mb"] > 100:
        value_score += 3
        reasons.append("Large model (>100MB)")
    elif file_info["size_mb"] > 10:
        value_score += 2

    # Checkpoints are less valuable than final models
    if not file_info["is_checkpoint"]:
        value_score += 2
        reasons.append("Final model (not checkpoint)")

    # Config files indicate well-documented models
    if file_info["config_files"]:
        value_score += 2
        reasons.append("Has config files")

    # Determine value level
    if value_score >= 6:
        value_level = "Critical"
    elif value_score >= 4:
        value_level = "High"
    elif value_score >= 2:
        value_level = "Medium"

    return {"value_score": value_score, "value_level": value_level, "reasons": reasons}


def identify_integration_opportunities(file_info: Dict, assessment: Dict, file_type: str) -> Dict:
    """Identify integration opportunities for a file."""
    opportunities = {
        "can_integrate": False,
        "integration_priority": "None",
        "integration_notes": [],
        "modifications_needed": [],
        "dependencies": [],
    }

    if file_type == "code":
        # Check if already in KmiDi
        path = file_info["path"]
        if "/KmiDi/" in path or "/KmiDi-1/" in path:
            opportunities["integration_notes"].append("Already in KmiDi project")
            return opportunities

        # High value files are good candidates
        if assessment["value_level"] in ["Critical", "High"]:
            opportunities["can_integrate"] = True
            opportunities["integration_priority"] = assessment["value_level"]

            if assessment["is_kmidi_related"]:
                opportunities["integration_notes"].append("KmiDi-related code - high priority")

            # Check for dependencies
            imports = file_info.get("imports", []) or file_info.get("includes", [])
            if imports:
                opportunities["dependencies"] = imports[:10]  # Limit to 10

        elif assessment["value_level"] == "Medium":
            opportunities["can_integrate"] = True
            opportunities["integration_priority"] = "Medium"
            opportunities["integration_notes"].append("May need adaptation")

    elif file_type == "documentation":
        if assessment["value_level"] in ["Critical", "High"]:
            opportunities["can_integrate"] = True
            opportunities["integration_priority"] = assessment["value_level"]
            opportunities["integration_notes"].append("Valuable documentation")

    elif file_type == "model":
        if assessment["value_level"] in ["Critical", "High"]:
            opportunities["can_integrate"] = True
            opportunities["integration_priority"] = assessment["value_level"]
            opportunities["integration_notes"].append("Valuable model")

            if file_info["config_files"]:
                opportunities["modifications_needed"].append("Include config files")

    return opportunities


def main():
    """Main analysis function."""
    print("Analyzing discovered files for value and integration opportunities...")
    print("=" * 60)

    discovery_dir = Path(__file__).parent.parent.parent / "docs" / "DISCOVERY"
    data = load_discovery_data(discovery_dir)

    if not data:
        print("No discovery data found. Run discovery scripts first.")
        return

    # Get KmiDi-related paths for context
    kmidi_paths = {"/KmiDi/", "/KmiDi-1/", "/kmidi", "/music_brain", "/penta_core"}

    results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "code_files": [],
        "documentation_files": [],
        "model_files": [],
        "summary": {
            "by_value_level": defaultdict(int),
            "integration_opportunities": defaultdict(int),
        },
    }

    # Analyze code files
    if "code_files" in data:
        print("Analyzing code files...")
        code_data = data["code_files"]
        for file_info in code_data.get("files", []):
            assessment = assess_code_file_value(file_info, kmidi_paths)
            opportunities = identify_integration_opportunities(file_info, assessment, "code")

            result = {
                "file": file_info["path"],
                "assessment": assessment,
                "opportunities": opportunities,
            }
            results["code_files"].append(result)
            results["summary"]["by_value_level"][assessment["value_level"]] += 1

            if opportunities["can_integrate"]:
                results["summary"]["integration_opportunities"][
                    opportunities["integration_priority"]
                ] += 1

    # Analyze documentation files
    if "documentation" in data:
        print("Analyzing documentation files...")
        doc_data = data["documentation"]
        for file_info in doc_data.get("files", []):
            assessment = assess_documentation_value(file_info)
            opportunities = identify_integration_opportunities(
                file_info, assessment, "documentation"
            )

            result = {
                "file": file_info["path"],
                "assessment": assessment,
                "opportunities": opportunities,
            }
            results["documentation_files"].append(result)
            results["summary"]["by_value_level"][assessment["value_level"]] += 1

            if opportunities["can_integrate"]:
                results["summary"]["integration_opportunities"][
                    opportunities["integration_priority"]
                ] += 1

    # Analyze model files
    if "models" in data:
        print("Analyzing model files...")
        model_data = data["models"]
        for file_info in model_data.get("files", []):
            assessment = assess_model_value(file_info)
            opportunities = identify_integration_opportunities(file_info, assessment, "model")

            result = {
                "file": file_info["path"],
                "assessment": assessment,
                "opportunities": opportunities,
            }
            results["model_files"].append(result)
            results["summary"]["by_value_level"][assessment["value_level"]] += 1

            if opportunities["can_integrate"]:
                results["summary"]["integration_opportunities"][
                    opportunities["integration_priority"]
                ] += 1

    # Convert defaultdicts to regular dicts
    results["summary"]["by_value_level"] = dict(results["summary"]["by_value_level"])
    results["summary"]["integration_opportunities"] = dict(
        results["summary"]["integration_opportunities"]
    )

    # Save results
    json_path = discovery_dir / "value_assessment.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved analysis results to: {json_path}")

    # Generate summary report
    md_path = discovery_dir / "value_categorization.md"
    with open(md_path, "w") as f:
        f.write("# Value Categorization and Integration Opportunities\n\n")
        f.write(f"**Analysis Date**: {results['analysis_timestamp']}\n\n")

        f.write("## Summary\n\n")
        f.write("### Files by Value Level\n\n")
        for level in ["Critical", "High", "Medium", "Low"]:
            count = results["summary"]["by_value_level"].get(level, 0)
            f.write(f"- **{level}**: {count}\n")
        f.write("\n")

        f.write("### Integration Opportunities\n\n")
        for priority in ["Critical", "High", "Medium"]:
            count = results["summary"]["integration_opportunities"].get(priority, 0)
            if count > 0:
                f.write(f"- **{priority} Priority**: {count} files\n")
        f.write("\n")

        # Top integration opportunities
        f.write("## Top Integration Opportunities\n\n")

        # Code files
        f.write("### High-Value Code Files\n\n")
        high_value_code = [
            r
            for r in results["code_files"]
            if r["assessment"]["value_level"] in ["Critical", "High"]
            and r["opportunities"]["can_integrate"]
        ]
        high_value_code.sort(key=lambda x: x["assessment"]["value_score"], reverse=True)

        for result in high_value_code[:50]:
            f.write(f"- **{result['file']}**\n")
            f.write(
                f"  - Value: {result['assessment']['value_level']} (Score: {result['assessment']['value_score']})\n"
            )
            f.write(f"  - Priority: {result['opportunities']['integration_priority']}\n")
            if result["assessment"]["reasons"]:
                f.write(f"  - Reasons: {', '.join(result['assessment']['reasons'])}\n")
            f.write("\n")

        # Documentation files
        f.write("### High-Value Documentation Files\n\n")
        high_value_docs = [
            r
            for r in results["documentation_files"]
            if r["assessment"]["value_level"] in ["Critical", "High"]
            and r["opportunities"]["can_integrate"]
        ]
        high_value_docs.sort(key=lambda x: x["assessment"]["value_score"], reverse=True)

        for result in high_value_docs[:30]:
            f.write(f"- **{result['file']}**\n")
            f.write(
                f"  - Value: {result['assessment']['value_level']} (Score: {result['assessment']['value_score']})\n"
            )
            if result["assessment"]["reasons"]:
                f.write(f"  - Reasons: {', '.join(result['assessment']['reasons'])}\n")
            f.write("\n")

        # Model files
        f.write("### High-Value Model Files\n\n")
        high_value_models = [
            r
            for r in results["model_files"]
            if r["assessment"]["value_level"] in ["Critical", "High"]
            and r["opportunities"]["can_integrate"]
        ]
        high_value_models.sort(key=lambda x: x["assessment"]["value_score"], reverse=True)

        for result in high_value_models[:20]:
            f.write(f"- **{result['file']}**\n")
            f.write(
                f"  - Value: {result['assessment']['value_level']} (Score: {result['assessment']['value_score']})\n"
            )
            if result["assessment"]["reasons"]:
                f.write(f"  - Reasons: {', '.join(result['assessment']['reasons'])}\n")
            f.write("\n")

    print(f"Saved value categorization report to: {md_path}")
    print("\n" + "=" * 60)
    print("Value assessment complete!")


if __name__ == "__main__":
    main()
