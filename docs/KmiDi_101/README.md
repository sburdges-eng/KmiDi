# KmiDi 101 — Course Module

Plain-language guide to the KmiDi project: what each part does, what depends on it, and how it ties together. Written for non-developers; uses real file and folder names.

## Contents

| File | Sections | Description |
| ---- | -------- | ----------- |
| [00_Overview.md](00_Overview.md) | 0–1 | What KmiDi is; how the pieces fit (big picture). |
| [02_Through_09_By_Area.md](02_Through_09_By_Area.md) | 2–9 | Brain, Intent, generate path, music_brain folders, Tauri, C++ engine, Web UI, build and run. |
| [10_Dependency_Map.md](10_Dependency_Map.md) | 10 | Who calls whom (main generate path and intent contract). |
| [11_Handoff.md](11_Handoff.md) | 11 | How to extend the 101 and pass the baton to the next writer/session. |
| [DISCOVERY_WORKFLOW.md](DISCOVERY_WORKFLOW.md) | — | How to find "what depends on X" and update the dependency map. When using the Cursor rule for dependency discovery, follow this workflow. |
| [KmiDi_101_NotebookLM_MindMap.md](KmiDi_101_NotebookLM_MindMap.md) | — | Hierarchy-only source for NotebookLM: upload to a notebook and ask for a **Mind Map** to get a branching overview of docs, generate path, and intent contract. |
| [KmiDi_101_Concatenated_MindMap.md](KmiDi_101_Concatenated_MindMap.md) | — | Single consolidated mind map (unified diagram + node → project file mapping); concatenates the NotebookLM views and links each node to repo files. |

### Combined map: docs, generate path, intent contract

```mermaid
flowchart TB
  subgraph docs["KmiDi 101 docs"]
    O["00_Overview: What KmiDi is + how pieces fit"]
    A["02_Through_09: Brain, Intent, path, music_brain, Tauri, C++, Web UI, build/run"]
    D["10_Dependency_Map: Who calls whom"]
    O --> A --> D
  end

  subgraph path["Generate path (who calls whom)"]
    IB["IntentBuilder"]
    Hook["useMusicBrain"]
    EP["POST /generate"]
    Handler["generate_music"]
    Process["process_song_intent"]
    Core["process_intent"]
    Engines["harmony, groove, kelly_companion..."]
    IB -->|generateFromIntent| Hook -->|POST| EP --> Handler --> Process --> Core --> Engines
  end

  subgraph intent["Intent contract (shared shape)"]
    Py["schema.py"]
    Sync["sync_entities.py"]
    JSON["CompleteSongIntentRequest.json"]
    TS["Intent.ts"]
    RS["intent.rs"]
    Py -->|source| Sync
    Sync -->|generates| JSON
    Sync -->|generates| TS
    Sync -->|generates| RS
  end

  D --> path
  D --> intent
  Py -.->|validates body| EP
  JSON -.-> TS
  JSON -.-> RS
```

Below, each process action is elaborated at file and code level; expand by scrolling to the next level.

### Level 1 — Generate path (file and code level)

```mermaid
flowchart TB
  subgraph webUi [Web UI]
    IB["IntentBuilder\nsrc/components/IntentBuilder.tsx"]
    BGP["buildGeneratePayload\nsrc/hooks/useMusicBrain.ts L121"]
    GM["generateMusic\nsrc/hooks/useMusicBrain.ts L153\nPOST body to API"]
    IB -->|"buildGeneratePayload(intent); generateMusic(payload)"| BGP
    BGP --> GM
  end

  subgraph api [Music Brain API]
    PostGen["POST /generate\nmusic_brain/api.py L1336"]
    GenMusic["generate_music\nmusic_brain/api.py L1337"]
    Validate["CompleteSongIntentRequest.model_validate\nmusic_brain/engine_api/schema.py"]
    Convert["_convert_to_intent\nmusic_brain/api.py L1403 local"]
    ProcSong["process_song_intent\nmusic_brain/api.py L699 DAiWAPI"]
    ProcIntent["process_intent\nmusic_brain/session/intent_processor.py L719"]
    GenAll["IntentProcessor.generate_all\nmusic_brain/session/intent_processor.py L702"]
    GenMusic --> Validate
    Validate --> Convert
    Convert --> ProcSong
    ProcSong --> ProcIntent
    ProcIntent --> GenAll
  end

  GM -->|"HTTP POST"| PostGen
  PostGen --> GenMusic
```

### Level 2 — process_intent expanded (file and code level)

```mermaid
flowchart TB
  PI["process_intent\nmusic_brain/session/intent_processor.py L719"]
  IP["IntentProcessor(intent)"]
  GA["generate_all\nmusic_brain/session/intent_processor.py L702"]
  GH["generate_harmony\nintent_processor.py IntentProcessor"]
  GG["generate_groove\nintent_processor.py IntentProcessor"]
  GArr["generate_arrangement\nintent_processor.py IntentProcessor"]
  GProd["generate_production\nintent_processor.py IntentProcessor"]
  Sum["intent_summary\nL708-714"]

  PI --> IP
  IP --> GA
  GA --> GH
  GA --> GG
  GA --> GArr
  GA --> GProd
  GA --> Sum
```

### Level 3 — Intent contract (file and code level)

```mermaid
flowchart LR
  subgraph source [Source]
    SchemaPy["schema.py\nmusic_brain/engine_api/schema.py\nCompleteSongIntentRequest Pydantic"]
  end

  subgraph sync [Sync script]
    SyncScript["sync_entities.py\nscripts/sync_entities.py\nreads CompleteSongIntentRequest"]
  end

  subgraph outputs [Generated outputs]
    JSONOut["CompleteSongIntentRequest.json\nshared_schemas/"]
    TSOut["Intent.ts\nsrc/types/"]
    RSOut["intent.rs\nsrc-tauri/src/generated/"]
  end

  SchemaPy -->|"source"| SyncScript
  SyncScript -->|"writes"| JSONOut
  SyncScript -->|"writes"| TSOut
  SyncScript -->|"writes"| RSOut
  SchemaPy -.->|"validates /generate body"| ApiPy["api.py generate_music"]
```

Read in order for a full pass, or jump to a section by area (2–9) or the dependency map (10).
