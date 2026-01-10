# DAiW Integration Architecture

## Where Does Gemini Research Go?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  GEMINI (Research)                                                          │
│  ───────────────────                                                        │
│  "Find examples of modal mixture"                                           │
│  "Research tension-emotion mapping"                                         │
│                     │                                                       │
│                     ▼                                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │  RESEARCH PACKET (Markdown)         │                                    │
│  │  - Sources cited                    │                                    │
│  │  - Examples found                   │                                    │
│  │  - Key findings                     │                                    │
│  └─────────────────────────────────────┘                                    │
│                     │                                                       │
│                     ▼                                                       │
│  CLAUDE CHAT (Synthesis & Design)                                           │
│  ─────────────────────────────────                                          │
│  - Synthesize into design decisions                                         │
│  - Create code structures                                                   │
│  - Update presets/mappings                                                  │
│                     │                                                       │
│                     ▼                                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │  CODE FILES                         │                                    │
│  │  - emotional_mapping.py             │◄──── Research becomes code         │
│  │  - rule_breaking_masterpieces.md    │◄──── Examples become reference     │
│  │  - EMOTIONAL_PRESETS dict           │◄──── Findings become presets       │
│  └─────────────────────────────────────┘                                    │
│                     │                                                       │
│                     ▼                                                       │
│  CLAUDE CODE / CURSOR (Implementation)                                      │
│  ─────────────────────────────────────                                      │
│  - Implement in music_brain package                                         │
│  - Write tests                                                              │
│  - Integrate with existing modules                                          │
│                     │                                                       │
│                     ▼                                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │  GITHUB REPO                        │                                    │
│  │  music_brain/models/emotional_mapping.py                                 │
│  │  music_brain/data/presets.json      │                                    │
│  │  docs/rule_breaking_masterpieces.md │                                    │
│  └─────────────────────────────────────┘                                    │
│                     │                                                       │
│                     ▼                                                       │
│  CUSTOM GPTs (Use the knowledge)                                            │
│  ───────────────────────────────                                            │
│  - Upload code files as knowledge                                           │
│  - Reference in system prompts                                              │
│  - GPT can now use the research                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Concrete Example: Emotion Mapping Research → Code → GPT

### Step 1: Gemini Research
```
You: "Research academic approaches to mapping emotional states to musical parameters"

Gemini outputs: Research packet with Russell's circumplex model, tempo ranges, etc.
```

### Step 2: Claude Synthesizes
```
Claude takes research → Creates emotional_mapping.py with:
- EmotionalState dataclass
- MusicalParameters dataclass  
- EMOTIONAL_PRESETS dictionary
- INTERVAL_EMOTIONS mapping
- get_parameters_for_state() function
```

### Step 3: Code Goes to Repo
```bash
# Claude Code or Cursor implements
cp emotional_mapping.py music_brain/models/
pytest tests/models/test_emotional_mapping.py
git add . && git commit -m "Add emotional mapping module"
```

### Step 4: GPT Gets Knowledge
```
Upload to Custom GPT:
- emotional_mapping.py (the code)
- Research packet (the sources)

Now GPT can reference: "Based on Russell's circumplex model in emotional_mapping.py..."
```

---

## File Locations in Your Repo

```
DAiW-Music-Brain/
├── music_brain/
│   ├── models/
│   │   ├── emotional_mapping.py    ◄── Research → Code
│   │   ├── emotional_intent.py
│   │   └── tension_curve.py
│   ├── harmony/
│   ├── groove/
│   └── data/
│       ├── presets.json            ◄── Extracted presets
│       └── interval_emotions.json  ◄── Extracted mappings
├── docs/
│   ├── research/
│   │   ├── emotion_mapping_research.md   ◄── Gemini packets saved
│   │   ├── modal_mixture_examples.md
│   │   └── lo_fi_production_research.md
│   └── rule_breaking_masterpieces.md
├── .claude/
│   └── settings.json
├── .cursorrules
└── CLAUDE.md
```

---

## Research Storage Format

When Gemini gives you a research packet, save it:

```markdown
# docs/research/[topic]_research.md

## Source: Gemini
## Date: 2025-11-27
## Query: "[your original question]"

---

[Paste Gemini's full response here]

---

## Integration Notes (added by Claude)
- Used to create: music_brain/models/emotional_mapping.py
- Key insights applied: [list]
- Gaps to fill: [list]
```

This creates an audit trail of where knowledge came from.

---

## Custom GPT Knowledge Integration

Your Custom GPT should have these files uploaded:

### Required Knowledge Files
1. **emotional_mapping.py** - The code with presets and mappings
2. **rule_breaking_masterpieces.md** - Theory reference
3. **Project README** - Context

### Optional (for deeper context)
4. **Research packets** - Gemini outputs
5. **Key module files** - From music_brain/

### In GPT Instructions, reference them:
```
When asked about emotional mapping, reference the EMOTIONAL_PRESETS 
dictionary in emotional_mapping.py. Use Russell's circumplex model 
(arousal x valence) as the foundation.

When asked about rule-breaking techniques, reference rule_breaking_masterpieces.md
for specific examples with notation details.
```
