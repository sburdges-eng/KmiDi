# Streamlit Cloud Deployment Guide

> Deploy iDAW web interface to Streamlit Cloud for public access.

## Prerequisites

1. GitHub repository with Streamlit app
2. Streamlit Cloud account (https://streamlit.io/cloud)
3. Python 3.9+

## Project Structure

```
iDAW/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit configuration
└── deployment/
    └── streamlit_cloud.md    # This file
```

## Deployment Steps

### 1. Create Streamlit App Entry Point

```python
# streamlit_app.py
import streamlit as st
from music_brain import IntentSchema, HarmonyGenerator, GrooveApplicator

st.set_page_config(
    page_title="iDAW - Intelligent DAW",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 iDAW - Intelligent Digital Audio Workstation")
st.markdown("*Interrogate Before Generate*")

# Intent input
with st.sidebar:
    st.header("Song Intent")
    mood = st.selectbox("Primary Mood", ["melancholic", "joyful", "angry", "peaceful"])
    key = st.selectbox("Key", ["C", "D", "E", "F", "G", "A", "B"])
    mode = st.selectbox("Mode", ["major", "minor", "dorian", "mixolydian"])

# Main content
if st.button("Generate Harmony"):
    with st.spinner("Generating..."):
        # Generate harmony based on intent
        result = generate_harmony(mood, key, mode)
        st.success("Harmony generated!")
        st.json(result)
```

### 2. Configure Streamlit

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#4A90D9"
backgroundColor = "#1a1a1a"
secondaryBackgroundColor = "#2d2d2d"
textColor = "#ffffff"
font = "monospace"

[server]
maxUploadSize = 50
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### 3. Create Requirements File

```
# requirements.txt
streamlit>=1.28.0
numpy>=1.24.0
mido>=1.3.0
pydantic>=2.0.0
```

### 4. Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Connect GitHub repository
4. Select:
   - Repository: `sburdges-eng/iDAW`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
5. Click "Deploy"

### 5. Configure Secrets (if needed)

In Streamlit Cloud dashboard, add secrets:

```toml
# .streamlit/secrets.toml (local development)
# In cloud: Dashboard > App settings > Secrets

[api]
openai_key = "sk-..."

[database]
connection_string = "..."
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `IDAW_DEBUG` | Enable debug mode | No |
| `IDAW_MODEL_PATH` | Path to ML models | No |

## Custom Domain

To use a custom domain:

1. Streamlit Cloud > App settings > Custom domain
2. Add domain: `app.idaw.dev`
3. Configure DNS CNAME record
4. Wait for SSL certificate provisioning

## Resource Limits

Streamlit Cloud free tier:

- 1 GB RAM
- 1 CPU core
- 1 GB storage

For larger deployments, consider:

- Streamlit Cloud Teams
- Self-hosted on Railway/Render
- Docker deployment

## Health Checks

Add a health endpoint:

```python
# Health check for load balancers
@st.cache_data
def health_check():
    return {"status": "healthy", "version": "1.0.0"}
```

## Monitoring

View logs in Streamlit Cloud dashboard:

- App logs
- Build logs
- Crash reports

## Troubleshooting

### App Won't Start
- Check requirements.txt for correct versions
- Verify Python version compatibility
- Check for import errors in logs

### Slow Performance
- Use `@st.cache_data` for expensive computations
- Reduce ML model size
- Use session state for user data

### Memory Issues
- Profile memory usage
- Lazy load large models
- Use generators instead of lists

## Alternative Platforms

### Railway
```bash
# railway.toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "streamlit run streamlit_app.py --server.port $PORT"
```

### Render
```yaml
# render.yaml
services:
  - type: web
    name: idaw
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run streamlit_app.py --server.port $PORT
```

### Heroku
```
# Procfile
web: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

## Support

- Streamlit Docs: https://docs.streamlit.io
- iDAW Issues: https://github.com/sburdges-eng/iDAW/issues

---

*"The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"*
