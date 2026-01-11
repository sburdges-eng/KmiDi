"""
Overlap analyzer
Identifies shared components across pipelines
"""

from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

class OverlapAnalyzer:
    """Analyzes shared components across pipelines"""
    
    def analyze_overlap(self, pipeline_components: Dict, 
                       unified_systems: List[Dict],
                       ui_ux_data: Dict) -> Dict:
        """Analyze overlap between pipelines"""
        
        # Find files used in multiple pipelines
        file_pipelines = defaultdict(set)
        
        for pipeline_type, components in pipeline_components.items():
            if pipeline_type == "shared":
                continue
            for comp in components:
                file_path = comp.get("path")
                if file_path:
                    file_pipelines[file_path].add(pipeline_type)
        
        # Find shared components
        shared_components_list = []
        for file_path, pipelines in file_pipelines.items():
            if len(pipelines) > 1:
                # Find component info
                component_info = None
                for pipeline_type in pipelines:
                    for comp in pipeline_components.get(pipeline_type, []):
                        if comp.get("path") == file_path:
                            component_info = comp
                            break
                    if component_info:
                        break
                
                if component_info:
                    shared_components_list.append({
                        "name": Path(file_path).stem,
                        "description": f"Shared component used in {', '.join(pipelines)}",
                        "used_in": list(pipelines),
                        "path": file_path,
                        "overlap_benefits": self._generate_overlap_benefits(pipelines)
                    })
        
        # Find shared UI components
        shared_ui = []
        ui_by_name = defaultdict(list)
        
        for pipeline_type, ui_data in ui_ux_data.items():
            if pipeline_type == "shared":
                continue
            for comp in ui_data.get("components", []):
                comp_name = comp.get("name")
                if comp_name:
                    ui_by_name[comp_name].append({
                        "pipeline": pipeline_type,
                        "component": comp
                    })
        
        for comp_name, occurrences in ui_by_name.items():
            if len(occurrences) > 1:
                pipelines = [occ["pipeline"] for occ in occurrences]
                shared_ui.append({
                    "name": comp_name,
                    "used_in": pipelines,
                    "components": [occ["component"] for occ in occurrences],
                    "overlap_benefits": self._generate_overlap_benefits(pipelines)
                })
        
        return {
            "components": shared_components_list,
            "ui_components": shared_ui,
            "overlap_summary": {
                "shared_code_components": len(shared_components_list),
                "shared_ui_components": len(shared_ui),
                "total_overlap": len(shared_components_list) + len(shared_ui)
            }
        }
    
    def _generate_overlap_benefits(self, pipelines: Set[str]) -> List[str]:
        """Generate benefits of component overlap"""
        benefits = [
            f"Consistent behavior across {', '.join(pipelines)}",
            "Reduced code duplication",
            "Easier maintenance and updates",
            "Unified API simplifies development"
        ]
        return benefits
