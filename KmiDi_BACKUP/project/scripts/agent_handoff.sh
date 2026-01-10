#!/bin/bash
# Agent handoff logging script

AGENT_NAME=$1
SUMMARY=$2
DATE=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE=".agents/logs/${AGENT_NAME}_${DATE}.md"

if [ -z "$AGENT_NAME" ] || [ -z "$SUMMARY" ]; then
    echo "Usage: ./agent_handoff.sh <agent_name> <summary>"
    echo "Example: ./agent_handoff.sh frontend \"Completed Emotion Wheel component\""
    exit 1
fi

cat > "$LOG_FILE" << HANDOFF
# Agent Handoff: ${AGENT_NAME}

**Date:** $(date +%Y-%m-%d)
**Time:** $(date +%H:%M:%S)
**Agent:** ${AGENT_NAME}

## Summary
${SUMMARY}

## Git Status
\`\`\`
$(git status --short)
\`\`\`

## Recent Commits
\`\`\`
$(git log --oneline -5)
\`\`\`

## Next Session Checklist
- [ ] Read CURRENT_STATE.md
- [ ] Review context file
- [ ] Continue from here
HANDOFF

echo "✅ Handoff logged: $LOG_FILE"
echo ""
echo "Don't forget to update .agents/handoffs/CURRENT_STATE.md!"
