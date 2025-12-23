# Kelly Project Multi-Agent Development System

## Active Agents

### Agent 1: Frontend Specialist (Cursor + Claude)
- **Primary:** Cursor AI in VS Code
- **Backup:** Claude for architecture decisions
- **Domain:** React, TypeScript, Tailwind, Side A/Side B UI
- **Context:** `.agents/contexts/frontend_context.md`

### Agent 2: Audio Engine Specialist (Claude)
- **Primary:** Claude
- **Domain:** Rust, C++, Tauri, penta-core, audio processing
- **Context:** `.agents/contexts/audio_context.md`

### Agent 3: Music Brain Specialist (Claude)
- **Primary:** Claude
- **Domain:** Python, music generation, emotion system, MIDI
- **Context:** `.agents/contexts/musicbrain_context.md`

### Agent 4: Documentation/DevOps (Gemini/ChatGPT)
- **Primary:** Gemini or ChatGPT
- **Domain:** Documentation, build scripts, testing
- **Context:** `.agents/contexts/devops_context.md`

## Handoff Protocol

When switching agents:
1. Update `.agents/handoffs/CURRENT_STATE.md`
2. Log work in `.agents/logs/[agent_name]_[date].md`
3. Update relevant context file
4. Next agent reads CURRENT_STATE.md first

## Quick Start Commands

### Starting work with an agent:
cat .agents/handoffs/CURRENT_STATE.md
cat .agents/contexts/[agent]_context.md

### Ending work session:
./agent_handoff.sh [agent_name] "Brief summary of what was done"
