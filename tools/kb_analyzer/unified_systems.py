"""
Unified system detection
Identifies components that work together as systems
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class UnifiedSystemDetector:
    """Detects unified systems (related components)"""
    
    def __init__(self):
        self.import_graph = defaultdict(set)
        self.file_dependencies = {}
    
    def detect_systems(self, files: List[Path], repo_dir: Path, 
                      categorized: Dict) -> List[Dict]:
        """Detect unified systems"""
        # Build import graph
        self._build_import_graph(files, repo_dir)
        
        # Find connected components (systems)
        systems = []
        visited = set()
        
        for file_path in files:
            if str(file_path) in visited:
                continue
            
            # Find all connected files
            connected = self._find_connected_files(file_path, visited)
            if len(connected) > 1:
                system = self._create_system(connected, repo_dir, categorized)
                if system:
                    systems.append(system)
        
        return systems
    
    def _build_import_graph(self, files: List[Path], repo_dir: Path):
        """Build graph of file imports"""
        for file_path in files:
            if file_path.suffix != ".py":
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                tree = ast.parse(content, filename=str(file_path))
                
                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
                
                self.file_dependencies[str(file_path)] = imports
                
                # Try to resolve imports to actual files
                for imp in imports:
                    # Simple heuristic: check if import matches a file name
                    for other_file in files:
                        if other_file.suffix == ".py":
                            module_name = other_file.stem
                            if imp == module_name or imp in str(other_file):
                                self.import_graph[str(file_path)].add(str(other_file))
                                self.import_graph[str(other_file)].add(str(file_path))
            
            except Exception:
                continue
    
    def _find_connected_files(self, start_file: Path, visited: Set) -> Set[Path]:
        """Find all files connected to start_file via imports"""
        connected = set()
        queue = [str(start_file)]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            connected.add(Path(current))
            
            # Add neighbors
            for neighbor in self.import_graph.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return connected
    
    def _create_system(self, files: Set[Path], repo_dir: Path, 
                      categorized: Dict) -> Dict:
        """Create a system description from connected files"""
        if len(files) < 2:
            return None
        
        # Determine system name from most common path component
        path_components = defaultdict(int)
        for file_path in files:
            parts = file_path.relative_to(repo_dir).parts
            if len(parts) > 0:
                path_components[parts[0]] += 1
        
        if path_components:
            system_name = max(path_components.items(), key=lambda x: x[1])[0]
            system_name = system_name.replace("_", " ").title() + " System"
        else:
            system_name = "Unified System"
        
        # Get categories
        categories = set()
        for file_path in files:
            rel_path = str(file_path.relative_to(repo_dir))
            for cat, files_list in categorized.items():
                for file_info in files_list:
                    if file_info.get("path") == rel_path:
                        for cat_info in file_info.get("categories", []):
                            categories.add(cat_info["category"])
        
        # Create system description
        system = {
            "name": system_name,
            "description": f"Unified system with {len(files)} connected components",
            "components": [
                {
                    "type": "file",
                    "path": str(f.relative_to(repo_dir)),
                    "absolute_path": str(f)
                }
                for f in files
            ],
            "topics": list(categories),
            "component_count": len(files)
        }
        
        return system
