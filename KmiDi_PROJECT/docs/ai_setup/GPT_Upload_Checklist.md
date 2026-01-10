# GPT Upload Checklist & Commands

## Files Ready for Upload

Download these from your outputs folder:

| # | File | Size | Purpose |
|---|------|------|---------|
| 1 | `daiw_knowledge_base.json` | ~15KB | All presets, mappings, intervals |
| 2 | `rule_breaking_database.json` | ~12KB | Every rule-break example |
| 3 | `emotional_mapping.py` | ~10KB | Code with functions |
| 4 | `DAiW_Cheat_Sheet.md` | ~3KB | Quick reference |

**Total: ~40KB** (well under GPT's limits)

---

## Upload Order (Matters for GPT Parsing)

1. **daiw_knowledge_base.json** — Primary reference
2. **rule_breaking_database.json** — Examples database
3. **emotional_mapping.py** — Code patterns
4. **DAiW_Cheat_Sheet.md** — Quick lookups

---

## GPT Creation Checklist

```
☐ Go to chat.openai.com
☐ Click profile → My GPTs → Create a GPT
☐ Name: "DAiW Music Brain"
☐ Description: "AI music production assistant..."
☐ Paste system prompt from Custom_GPT_Build_Script.md
☐ Upload files (4 files above)
☐ Enable: Web Browsing, Code Interpreter
☐ Disable: DALL-E
☐ Add conversation starters
☐ Save
☐ Test with: "Help me map grief + anger to parameters"
```

---

## Test Prompts After Creation

Run these to verify GPT works:

**Test 1 - Preset Lookup:**
```
What are the parameters for anxiety?
```
Expected: Should cite 100-140 BPM, Phrygian, ahead of beat, 60% dissonance

**Test 2 - Rule Breaking:**
```
When should I use parallel fifths?
```
Expected: Should mention Beethoven, power chords, folk aesthetic

**Test 3 - Kelly Song:**
```
How do I apply misdirection to the Kelly song?
```
Expected: Should know F-C-Am-Dm, 82 BPM, surface love / undertow grief

**Test 4 - Interrogation:**
```
I feel anxious but also exhausted. Help me map that.
```
Expected: Should ask clarifying questions, suggest compound mapping

---

## Claude Code Command to Deploy to Repo

After GPT is working, put files in your repo:

```bash
claude "Deploy DAiW knowledge files:
1. Create music_brain/data/ if not exists
2. Copy daiw_knowledge_base.json to music_brain/data/presets.json
3. Copy rule_breaking_database.json to music_brain/data/rule_breaks.json  
4. Copy emotional_mapping.py to music_brain/models/emotional_mapping.py
5. Copy DAiW_Cheat_Sheet.md to docs/cheat_sheet.md
6. Update music_brain/__init__.py to import emotional_mapping
7. Create basic test file tests/models/test_emotional_mapping.py
8. Run pytest to verify
9. Show me the updated structure"
```

---

## Gemini Research Flow

When Gemini gives you research:

```bash
# Save the research packet
echo "[paste Gemini output]" > docs/research/[topic]_research.md

# Then tell Claude Code to integrate
claude "Read docs/research/[topic]_research.md and:
1. Extract any new presets → add to music_brain/data/presets.json
2. Extract any new examples → add to music_brain/data/rule_breaks.json
3. Update emotional_mapping.py if needed
4. Show me what changed"
```

Then re-upload updated JSON files to your Custom GPT.

---

## Full System Status Check

```
┌─────────────────────────────────────────────────┐
│ TOOL          │ STATUS    │ HAS CONTEXT?        │
├───────────────┼───────────┼─────────────────────┤
│ Claude Chat   │ ✅ Ready  │ ✅ (this convo)     │
│ Claude Code   │ ⏳ Setup  │ Run setup command   │
│ Cursor        │ ⏳ Setup  │ Install + .cursorrules │
│ Custom GPT    │ ⏳ Create │ Upload 4 files      │
│ Gemini        │ ✅ Ready  │ Paste system prompt │
└───────────────┴───────────┴─────────────────────┘
```

---

## Quick Links

- Claude Code install: `npm install -g @anthropic-ai/claude-code`
- Cursor download: https://cursor.com
- Cursor student free year: https://cursor.com/students
- ChatGPT GPT builder: https://chat.openai.com → My GPTs → Create
- Your repo: https://github.com/seanburdgeseng/DAiW-Music-Brain
