"""
Knowledge base builder
Creates structured knowledge base output
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class KnowledgeBaseBuilder:
    """Builds structured knowledge base"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build(self, categorized: Dict, pipeline_components: Dict,
              unified_systems: List[Dict], ui_ux_data: Dict,
              shared_components: Dict, pipeline_filter: str = "all",
              output_format: str = "both"):
        """Build knowledge base structure"""
        
        # Create directory structure
        pipelines_dir = self.output_dir / "pipelines"
        topics_dir = self.output_dir / "topics"
        
        # Create pipeline directories (singular keys: plugin, daw, standalone)
        for pipeline in ["plugin", "daw", "standalone"]:
            (pipelines_dir / pipeline / "ui_ux").mkdir(parents=True, exist_ok=True)
            (pipelines_dir / pipeline / "topics").mkdir(parents=True, exist_ok=True)
        
        # Build unified systems
        self._build_unified_systems(unified_systems)
        
        # Build shared components
        self._build_shared_components(shared_components)
        
        # Build pipeline-specific knowledge bases
        if pipeline_filter in ["all", "plugins"]:
            self._build_pipeline_kb("plugin", pipeline_components.get("plugin", []),
                                  ui_ux_data.get("plugin", {}), categorized)
        
        if pipeline_filter in ["all", "daw"]:
            self._build_pipeline_kb("daw", pipeline_components.get("daw", []),
                                  ui_ux_data.get("daw", {}), categorized)
        
        if pipeline_filter in ["all", "standalone"]:
            self._build_pipeline_kb("standalone", pipeline_components.get("standalone", []),
                                  ui_ux_data.get("standalone", {}), categorized)
        
        # Build topic-based knowledge base
        self._build_topics_kb(categorized, pipeline_components)
        
        # Build index
        self._build_index(unified_systems, shared_components, pipeline_components)
    
    def _build_unified_systems(self, unified_systems: List[Dict]):
        """Build unified systems JSON"""
        output = {
            "generated_at": datetime.now().isoformat(),
            "systems": unified_systems
        }
        
        output_path = self.output_dir / "unified_systems.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
    
    def _build_shared_components(self, shared_components: Dict):
        """Build shared components JSON"""
        output = {
            "generated_at": datetime.now().isoformat(),
            "shared_components": shared_components.get("components", [])
        }
        
        output_path = self.output_dir / "shared_components.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
    
    def _build_pipeline_kb(self, pipeline_type: str, components: List[Dict],
                          ui_ux_data: Dict, categorized: Dict):
        """Build pipeline-specific knowledge base"""
        pipeline_dir = self.output_dir / "pipelines" / pipeline_type
        
        # Components
        components_output = {
            "pipeline": pipeline_type,
            "generated_at": datetime.now().isoformat(),
            "components": components
        }
        
        with open(pipeline_dir / "components.json", 'w') as f:
            json.dump(components_output, f, indent=2)
        
        # Systems (filter unified systems for this pipeline)
        systems = []
        # This would need to filter unified_systems by pipeline
        systems_output = {
            "pipeline": pipeline_type,
            "generated_at": datetime.now().isoformat(),
            "systems": systems
        }
        
        with open(pipeline_dir / "systems.json", 'w') as f:
            json.dump(systems_output, f, indent=2)
        
        # UI/UX
        ui_ux_dir = pipeline_dir / "ui_ux"
        ui_ux_output = {
            "pipeline": pipeline_type,
            "generated_at": datetime.now().isoformat(),
            "components": ui_ux_data.get("components", []),
            "workflows": ui_ux_data.get("workflows", []),
            "design_patterns": ui_ux_data.get("design_patterns", [])
        }
        
        with open(ui_ux_dir / "components.json", 'w') as f:
            json.dump(ui_ux_output, f, indent=2)
    
    def _build_topics_kb(self, categorized: Dict, pipeline_components: Dict):
        """Build topic-based knowledge base"""
        topics_dir = self.output_dir / "topics"
        
        for topic, files in categorized.items():
            if topic == "uncategorized":
                continue
            
            topic_dir = topics_dir / topic
            topic_dir.mkdir(parents=True, exist_ok=True)
            
            # Tag files by pipeline
            for file_info in files:
                file_path = file_info["path"]
                pipelines = []
                
                for pipeline_type, components in pipeline_components.items():
                    for comp in components:
                        if comp.get("path") == file_path:
                            pipelines.append(pipeline_type)
                
                file_info["pipelines"] = pipelines
            
            output = {
                "topic": topic,
                "generated_at": datetime.now().isoformat(),
                "files": files,
                "file_count": len(files)
            }
            
            with open(topic_dir / "index.json", 'w') as f:
                json.dump(output, f, indent=2)
    
    def _build_index(self, unified_systems: List[Dict], shared_components: Dict,
                    pipeline_components: Dict):
        """Build main index"""
        index = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "unified_systems_count": len(unified_systems),
                "shared_components_count": len(shared_components.get("components", [])),
                "plugin_components": len(pipeline_components.get("plugin", [])),
                "daw_components": len(pipeline_components.get("daw", [])),
                "standalone_components": len(pipeline_components.get("standalone", []))
            },
            "structure": {
                "unified_systems": "unified_systems.json",
                "shared_components": "shared_components.json",
                "pipelines": {
                    "plugins": "pipelines/plugins/",
                    "daw": "pipelines/daw/",
                    "standalone": "pipelines/standalone/"
                },
                "topics": "topics/"
            }
        }
        
        with open(self.output_dir / "index.json", 'w') as f:
            json.dump(index, f, indent=2)
