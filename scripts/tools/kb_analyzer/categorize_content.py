"""
Content categorization logic
Identifies music-related topics and categories
"""

import re
from pathlib import Path
from typing import Dict, List, Set
import json

class Categorizer:
    """Categorizes files by music-related topics"""
    
    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "categories.json"
        
        # Load categories configuration
        if config_path.exists():
            with open(config_path) as f:
                self.categories = json.load(f)
        else:
            # Default categories
            self.categories = {
                "production": {
                    "keywords": ["mixing", "mastering", "recording", "audio", "sound", "mix", "compressor", "eq", "reverb"],
                    "patterns": [r"mix", r"master", r"record", r"audio"]
                },
                "theory": {
                    "keywords": ["chord", "scale", "harmony", "melody", "rhythm", "progression", "key", "mode"],
                    "patterns": [r"chord", r"scale", r"harmony", r"melody", r"progression"]
                },
                "ml_ai": {
                    "keywords": ["neural", "model", "training", "inference", "pytorch", "tensorflow", "ml", "ai", "transformer"],
                    "patterns": [r"model", r"train", r"neural", r"ml", r"ai"]
                },
                "audio_processing": {
                    "keywords": ["dsp", "fft", "filter", "synthesis", "effect", "processor", "audio"],
                    "patterns": [r"dsp", r"fft", r"filter", r"synthesis"]
                },
                "daw_integration": {
                    "keywords": ["logic", "ableton", "reaper", "daw", "plugin", "vst", "au"],
                    "patterns": [r"logic", r"ableton", r"reaper", r"daw"]
                },
                "instruments": {
                    "keywords": ["guitar", "piano", "drum", "synth", "bass", "violin"],
                    "patterns": [r"guitar", r"piano", r"drum", r"synth"]
                },
                "composition": {
                    "keywords": ["songwriting", "arrangement", "structure", "song", "composition"],
                    "patterns": [r"song", r"arrange", r"compose"]
                },
                "emotion": {
                    "keywords": ["emotion", "feeling", "valence", "arousal", "therapeutic"],
                    "patterns": [r"emotion", r"feeling", r"valence"]
                },
                "tools": {
                    "keywords": ["script", "utility", "tool", "helper", "automation"],
                    "patterns": [r"tool", r"script", r"util"]
                }
            }
    
    def categorize_files(self, files: List[Path], repo_dir: Path) -> Dict[str, List[Dict]]:
        """Categorize files by topic"""
        categorized = {category: [] for category in self.categories.keys()}
        categorized["uncategorized"] = []
        
        for file_path in files:
            # Skip binary and non-text files
            if not self._is_text_file(file_path):
                continue
            
            content = self._read_file_safe(file_path)
            if not content:
                continue
            
            file_info = {
                "path": str(file_path.relative_to(repo_dir)),
                "absolute_path": str(file_path),
                "categories": []
            }
            
            # Check each category
            for category, config in self.categories.items():
                score = self._calculate_category_score(content, file_path, config)
                if score > 0:
                    file_info["categories"].append({
                        "category": category,
                        "score": score
                    })
            
            # Add to appropriate categories
            if file_info["categories"]:
                # Sort by score
                file_info["categories"].sort(key=lambda x: x["score"], reverse=True)
                # Add to top category
                top_category = file_info["categories"][0]["category"]
                categorized[top_category].append(file_info)
            else:
                categorized["uncategorized"].append(file_info)
        
        return categorized
    
    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is a text file"""
        text_extensions = {".py", ".md", ".txt", ".json", ".yaml", ".yml", 
                          ".cpp", ".h", ".hpp", ".c", ".js", ".ts", ".tsx",
                          ".swift", ".rs", ".html", ".css", ".sh"}
        return file_path.suffix in text_extensions
    
    def _read_file_safe(self, file_path: Path, max_size: int = 1024 * 1024) -> str:
        """Safely read file content"""
        try:
            if file_path.stat().st_size > max_size:
                return ""  # Skip very large files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return ""
    
    def _calculate_category_score(self, content: str, file_path: Path, config: Dict) -> float:
        """Calculate how well a file matches a category"""
        score = 0.0
        content_lower = content.lower()
        path_lower = str(file_path).lower()
        
        # Check keywords
        keywords = config.get("keywords", [])
        for keyword in keywords:
            if keyword in content_lower:
                score += 1.0
            if keyword in path_lower:
                score += 2.0  # Higher weight for path matches
        
        # Check patterns
        patterns = config.get("patterns", [])
        for pattern in patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                score += 1.5
            if re.search(pattern, path_lower, re.IGNORECASE):
                score += 2.5
        
        return score
