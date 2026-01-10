import re
import os
import yaml
from pathlib import Path

ROADMAP_FILES = [
    "DAiW-Music-Brain/vault/Production_Workflows/hybrid_development_roadmap.md",
    "ROADMAP_penta-core.md"
]

class RoadmapObjective:
    def __init__(self, title, status, file_path, line_num):
        self.title = title
        self.status = status
        self.file_path = file_path
        self.line_num = line_num

    def to_dict(self):
        return {
            "title": self.title,
            "status": self.status,
            "file_path": self.file_path,
            "line": self.line_num,
        }

class RoadmapScanner:
    STATUS_RE = re.compile(r"\|\s*(.*?)\s*\|\s*(Working|TODO|Implemented|In Progress)\s*\|\s*`([^`]+)`")

    def scan(self):
        objectives = []
        for f in ROADMAP_FILES:
            path = Path(f)
            if not path.exists():
                continue
            with open(path, "r") as fh:
                for i, line in enumerate(fh):
                    m = self.STATUS_RE.search(line)
                    if m:
                        subsystem, status, file_ref = m.groups()
                        objectives.append(
                            RoadmapObjective(
                                title=subsystem.strip(),
                                status=status.strip(),
                                file_path=file_ref.strip(),
                                line_num=i + 1,
                            )
                        )
        return objectives
