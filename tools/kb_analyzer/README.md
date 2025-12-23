# Knowledge Base Analyzer

Analyzes the repository and creates a structured knowledge base organized by:
- Pipeline types (Plugins, DAW, Standalone)
- UI/UX components for each pipeline
- Topics and categories
- Unified systems
- Shared components across pipelines

## Usage

### Docker (Recommended)

```bash
# Run full analysis
docker-compose run --rm daiw-kb-analyzer

# Analyze specific pipeline
docker-compose run --rm -e KB_PIPELINE_FILTER=plugins daiw-kb-analyzer

# Analyze with UI/UX
docker-compose run --rm -e KB_INCLUDE_UI_UX=true daiw-kb-analyzer

# Analyze all pipelines
docker-compose run --rm -e KB_PIPELINE_FILTER=all -e KB_INCLUDE_UI_UX=true daiw-kb-analyzer
```

### Local

```bash
cd tools/kb_analyzer
python analyze_repository.py --repo-dir ../../ --output-dir ../../output/knowledge_base
```

## Output Structure

```
output/knowledge_base/
├── index.json                    # Main index
├── unified_systems.json          # All unified systems
├── shared_components.json        # Components shared across pipelines
├── pipelines/
│   ├── plugins/
│   │   ├── components.json
│   │   ├── systems.json
│   │   └── ui_ux/
│   │       └── components.json
│   ├── daw/
│   │   └── ...
│   └── standalone/
│       └── ...
└── topics/
    ├── production/
    ├── theory/
    └── ...
```

## Configuration

Edit `config/` files to customize:
- `categories.json` - Music topic categories
- `pipeline_patterns.json` - Pipeline detection patterns
- `ui_ux_patterns.json` - UI/UX framework patterns
