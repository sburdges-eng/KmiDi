"""
Pipeline detection logic
Identifies Plugin, DAW, and Standalone App components
"""

import re
from pathlib import Path
from typing import Dict, List, Set
import json

class PipelineDetector:
    """Detects pipeline types (Plugin, DAW, Standalone)"""
    
    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "pipeline_patterns.json"
        
        # Load pipeline patterns
        if config_path.exists():
            with open(config_path) as f:
                self.patterns = json.load(f)
        else:
            # Default patterns
            self.patterns = {
                "plugin": {
                    "file_patterns": [r"PluginProcessor", r"PluginEditor", r"AudioProcessor", r"\.vst3", r"\.component", r"\.clap"],
                    "path_patterns": [r"plugin", r"vst", r"au", r"juce"],
                    "keywords": ["juce::AudioProcessor", "VST3", "AU", "CLAP", "plugin"]
                },
                "daw": {
                    "file_patterns": [r"SessionManager", r"ProjectFile", r"DAWEngine", r"Session"],
                    "path_patterns": [r"daw", r"session", r"project"],
                    "keywords": ["session", "project", "daw", "arrangement", "timeline"]
                },
                "standalone": {
                    "file_patterns": [r"main\.swift", r"AppDelegate", r"Tauri", r"Electron", r"App\.tsx", r"App\.jsx"],
                    "path_patterns": [r"tauri", r"electron", r"src-tauri", r"macOS", r"iOS", r"mobile"],
                    "keywords": ["tauri", "electron", "desktop", "mobile", "app"]
                }
            }
    
    def detect_pipelines(self, files: List[Path], repo_dir: Path) -> Dict[str, List[Dict]]:
        """Detect which pipeline each file belongs to"""
        pipeline_components = {
            "plugin": [],
            "daw": [],
            "standalone": [],
            "shared": []
        }
        
        file_pipeline_scores = {}
        
        for file_path in files:
            if not self._is_code_file(file_path):
                continue
            
            content = self._read_file_safe(file_path)
            if not content:
                continue
            
            scores = {}
            for pipeline_type, config in self.patterns.items():
                score = self._calculate_pipeline_score(content, file_path, config)
                scores[pipeline_type] = score
            
            # Determine pipeline(s)
            max_score = max(scores.values()) if scores else 0
            if max_score > 0:
                pipelines = [p for p, s in scores.items() if s == max_score and s > 2.0]
                
                file_info = {
                    "path": str(file_path.relative_to(repo_dir)),
                    "absolute_path": str(file_path),
                    "pipelines": pipelines,
                    "scores": scores
                }
                
                if len(pipelines) > 1:
                    # Shared component
                    pipeline_components["shared"].append(file_info)
                elif len(pipelines) == 1:
                    pipeline_components[pipelines[0]].append(file_info)
            
            file_pipeline_scores[str(file_path)] = scores
        
        return pipeline_components
    
    def _is_code_file(self, file_path: Path) -> bool:
        """Check if file is a code file"""
        code_extensions = {".py", ".cpp", ".h", ".hpp", ".c", ".js", ".ts", 
                          ".tsx", ".swift", ".rs", ".java", ".kt"}
        return file_path.suffix in code_extensions
    
    def _read_file_safe(self, file_path: Path, max_size: int = 512 * 1024) -> str:
        """Safely read file content"""
        try:
            if file_path.stat().st_size > max_size:
                return ""  # Skip very large files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return ""
    
    def _calculate_pipeline_score(self, content: str, file_path: Path, config: Dict) -> float:
        """Calculate how well a file matches a pipeline"""
        score = 0.0
        content_lower = content.lower()
        path_lower = str(file_path).lower()
        
        # Check file patterns
        file_patterns = config.get("file_patterns", [])
        for pattern in file_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                score += 3.0
            if re.search(pattern, path_lower, re.IGNORECASE):
                score += 4.0
        
        # Check path patterns
        path_patterns = config.get("path_patterns", [])
        for pattern in path_patterns:
            if re.search(pattern, path_lower, re.IGNORECASE):
                score += 2.0
        
        # Check keywords
        keywords = config.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in content_lower:
                score += 1.0
        
        return score
