"""
UI/UX knowledge extraction
Extracts UI components and design patterns for each pipeline
"""

import re
from pathlib import Path
from typing import Dict, List, Set
import json

class UIUXExtractor:
    """Extracts UI/UX components and patterns"""
    
    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "ui_ux_patterns.json"
        
        # Load UI/UX patterns
        if config_path.exists():
            with open(config_path) as f:
                self.patterns = json.load(f)
        else:
            # Default patterns
            self.patterns = {
                "react": {
                    "extensions": [".tsx", ".jsx"],
                    "patterns": [r"React\.Component", r"function\s+\w+\(.*\)\s*\{", r"const\s+\w+\s*=\s*\(.*\)\s*=>"],
                    "keywords": ["useState", "useEffect", "props", "component"]
                },
                "swiftui": {
                    "extensions": [".swift"],
                    "patterns": [r"@ViewBuilder", r"SwiftUI\.View", r"struct\s+\w+.*View"],
                    "keywords": ["View", "Button", "Text", "VStack", "HStack"]
                },
                "juce_ui": {
                    "extensions": [".cpp", ".h"],
                    "patterns": [r"juce::Component", r"juce::Slider", r"juce::Button", r"juce::Label"],
                    "keywords": ["Component", "Slider", "Button", "Label", "JUCE"]
                },
                "web": {
                    "extensions": [".html", ".css", ".js"],
                    "patterns": [r"<div", r"<button", r"class=", r"id="],
                    "keywords": ["html", "css", "javascript"]
                }
            }
    
    def extract_ui_ux(self, files: List[Path], repo_dir: Path, 
                      pipeline_components: Dict) -> Dict:
        """Extract UI/UX components for each pipeline"""
        ui_ux_data = {
            "plugin": {"components": [], "workflows": [], "design_patterns": []},
            "daw": {"components": [], "workflows": [], "design_patterns": []},
            "standalone": {"components": [], "workflows": [], "design_patterns": []},
            "shared": {"components": [], "workflows": [], "design_patterns": []}
        }
        
        # Extract UI components
        for pipeline_type in ["plugin", "daw", "standalone", "shared"]:
            pipeline_files = pipeline_components.get(pipeline_type, [])
            
            for file_info in pipeline_files:
                file_path = Path(file_info["absolute_path"])
                ui_components = self._extract_ui_components(file_path, repo_dir)
                
                if ui_components:
                    ui_ux_data[pipeline_type]["components"].extend(ui_components)
        
        return ui_ux_data
    
    def _extract_ui_components(self, file_path: Path, repo_dir: Path) -> List[Dict]:
        """Extract UI components from a file"""
        if not self._is_ui_file(file_path):
            return []
        
        content = self._read_file_safe(file_path)
        if not content:
            return []
        
        components = []
        
        # Detect UI framework
        ui_framework = self._detect_ui_framework(content, file_path)
        
        if ui_framework:
            # Extract component definitions
            if ui_framework == "react":
                components.extend(self._extract_react_components(content, file_path, repo_dir))
            elif ui_framework == "swiftui":
                components.extend(self._extract_swiftui_components(content, file_path, repo_dir))
            elif ui_framework == "juce_ui":
                components.extend(self._extract_juce_components(content, file_path, repo_dir))
            elif ui_framework == "web":
                components.extend(self._extract_web_components(content, file_path, repo_dir))
        
        return components
    
    def _is_ui_file(self, file_path: Path) -> bool:
        """Check if file is a UI file"""
        ui_extensions = {".tsx", ".jsx", ".swift", ".html", ".css", ".cpp", ".h"}
        return file_path.suffix in ui_extensions
    
    def _read_file_safe(self, file_path: Path, max_size: int = 512 * 1024) -> str:
        """Safely read file content"""
        try:
            if file_path.stat().st_size > max_size:
                return ""
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return ""
    
    def _detect_ui_framework(self, content: str, file_path: Path) -> str:
        """Detect which UI framework is used"""
        content_lower = content.lower()
        
        for framework, config in self.patterns.items():
            keywords = config.get("keywords", [])
            if any(kw.lower() in content_lower for kw in keywords):
                if file_path.suffix in config.get("extensions", []):
                    return framework
        
        return None
    
    def _extract_react_components(self, content: str, file_path: Path, repo_dir: Path) -> List[Dict]:
        """Extract React components"""
        components = []
        
        # Find function components
        func_pattern = r"(?:export\s+)?(?:const|function)\s+(\w+)\s*[=:]?\s*(?:\([^)]*\)\s*=>|\([^)]*\)\s*\{)"
        for match in re.finditer(func_pattern, content):
            component_name = match.group(1)
            if component_name[0].isupper():  # React components start with uppercase
                components.append({
                    "name": component_name,
                    "type": "react_component",
                    "file": str(file_path.relative_to(repo_dir)),
                    "framework": "react"
                })
        
        return components
    
    def _extract_swiftui_components(self, content: str, file_path: Path, repo_dir: Path) -> List[Dict]:
        """Extract SwiftUI views"""
        components = []
        
        # Find struct views
        struct_pattern = r"struct\s+(\w+)\s*:\s*View"
        for match in re.finditer(struct_pattern, content):
            view_name = match.group(1)
            components.append({
                "name": view_name,
                "type": "swiftui_view",
                "file": str(file_path.relative_to(repo_dir)),
                "framework": "swiftui"
            })
        
        return components
    
    def _extract_juce_components(self, content: str, file_path: Path, repo_dir: Path) -> List[Dict]:
        """Extract JUCE UI components"""
        components = []
        
        # Find Component classes
        class_pattern = r"class\s+(\w+)\s*:\s*(?:public\s+)?juce::Component"
        for match in re.finditer(class_pattern, content):
            component_name = match.group(1)
            components.append({
                "name": component_name,
                "type": "juce_component",
                "file": str(file_path.relative_to(repo_dir)),
                "framework": "juce"
            })
        
        return components
    
    def _extract_web_components(self, content: str, file_path: Path, repo_dir: Path) -> List[Dict]:
        """Extract web UI components"""
        components = []
        
        # Find HTML elements with IDs or classes (potential components)
        id_pattern = r'id=["\'](\w+)["\']'
        class_pattern = r'class=["\']([^"\']+)["\']'
        
        for match in re.finditer(id_pattern, content):
            component_id = match.group(1)
            components.append({
                "name": component_id,
                "type": "web_component",
                "file": str(file_path.relative_to(repo_dir)),
                "framework": "web"
            })
        
        return components
