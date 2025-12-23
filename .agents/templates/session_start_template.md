# Session Start Checklist

## Before You Begin

### 1. Read Current State
```bash
cat .agents/handoffs/CURRENT_STATE.md
```

### 2. Read Your Context
```bash
cat .agents/contexts/[your_agent]_context.md
```

### 3. Check Recent Logs
```bash
ls -lt .agents/logs/ | head -5
cat .agents/logs/[most_recent_log].md
```

### 4. Pull Latest Code
```bash
git pull origin main
```

### 5. Check for Conflicts with Other Agents
- Are you waiting on another agent?
- Do other agents need your work first?

## During Your Session

### Keep Notes
- Document decisions
- Note any blockers
- Track time spent

### Commit Frequently
```bash
git add .
git commit -m "[Agent Name] Brief description"
```

## End of Session

### 1. Update Current State
```bash
vim .agents/handoffs/CURRENT_STATE.md
```

### 2. Log Your Handoff
```bash
./agent_handoff.sh [agent_name] "What you accomplished"
```

### 3. Push Your Work
```bash
git push origin main
```

### 4. Note Next Steps
- What should you do next session?
- What do other agents need to know?
