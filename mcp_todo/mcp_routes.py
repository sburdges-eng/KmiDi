from mcp_roadmap import RoadmapScanner
from mcp_task_manager import TaskManager

def get_tasks():
    tm = TaskManager()
    return tm.list_tasks()

def resync_roadmap():
    scanner = RoadmapScanner()
    objectives = scanner.scan()

    tm = TaskManager()
    tm.add_tasks(objectives)
    return tm.list_tasks()

def get_incomplete():
    tm = TaskManager()
    return tm.incomplete()

def mark_complete(title):
    tm = TaskManager()
    tm.update_task_status(title, "Implemented")
    return {"status": "ok", "updated": title}
