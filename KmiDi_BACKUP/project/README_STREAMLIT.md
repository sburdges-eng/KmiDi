# KmiDi Streamlit Demo

A beautiful, user-friendly web interface for emotion-driven music generation using Streamlit.

## Features

- 🎭 **Emotion Presets** - Quick selection of predefined emotions
- 💭 **Emotional Intent Input** - Describe what you're feeling
- 🎹 **Technical Parameters** - Control key, tempo (BPM), and genre
- 🎵 **AI Music Generation** - Generate MIDI files from emotional intent
- 📥 **MIDI Download** - Download generated music files
- 📜 **Generation History** - Track your recent generations
- 🌐 **Cloud Ready** - Deploy to Streamlit Cloud, Heroku, or any platform

## Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install streamlit
   pip install -r requirements.txt
   ```

2. **Run the App**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open in Browser**
   - The app will automatically open at `http://localhost:8501`

### Using with FastAPI Service

1. **Start the FastAPI service** (optional)
   ```bash
   cd api
   uvicorn main:app --reload
   ```

2. **Configure API connection**
   - Set environment variable: `KMIDI_USE_API=true`
   - Set API URL: `KMIDI_API_URL=http://127.0.0.1:8000`
   - Or edit `.streamlit/secrets.toml`:
     ```toml
     KMIDI_API_URL = "http://127.0.0.1:8000"
     KMIDI_USE_API = "true"
     ```

3. **Run Streamlit**
   ```bash
   streamlit run streamlit_app.py
   ```

## Cloud Deployment

### Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit demo"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Configure Secrets** (if needed)
   - In Streamlit Cloud dashboard, go to Settings → Secrets
   - Add your secrets:
     ```toml
     KMIDI_API_URL = "https://your-api-url.com"
     KMIDI_USE_API = "true"
     ```

### Heroku

1. **Create Heroku App**
   ```bash
   heroku create kmidi-streamlit-demo
   ```

2. **Set Environment Variables**
   ```bash
   heroku config:set KMIDI_USE_API=false
   ```

3. **Deploy**
   ```bash
   git push heroku main
   ```

4. **Open**
   ```bash
   heroku open
   ```

### Docker

1. **Build Docker Image**
   ```bash
   docker build -t kmidi-streamlit -f Dockerfile.streamlit .
   ```

2. **Run Container**
   ```bash
   docker run -p 8501:8501 kmidi-streamlit
   ```

## Configuration

### Environment Variables

- `KMIDI_API_URL` - FastAPI service URL (default: `http://127.0.0.1:8000`)
- `KMIDI_USE_API` - Use FastAPI service instead of direct API (default: `false`)

### Secrets File

Create `.streamlit/secrets.toml`:

```toml
KMIDI_API_URL = "http://your-api-url.com"
KMIDI_USE_API = "true"
```

## Usage

1. **Enter Emotional Intent**
   - Type what you're feeling: "I'm feeling grief hidden as love"
   - Or select from emotion presets in the sidebar

2. **Adjust Parameters** (Optional)
   - **Key**: Musical key (or Auto)
   - **Tempo (BPM)**: Speed of the music
   - **Genre**: Musical style

3. **Generate Music**
   - Click "Generate Music" button
   - Wait for AI to create your composition

4. **Download MIDI**
   - Click "Download MIDI File" button
   - Import into your DAW or MIDI player

## Architecture

```
Streamlit Frontend (streamlit_app.py)
    ↓
FastAPI Service (api/main.py) [Optional]
    ↓
Music Brain API (music_brain/)
    ↓
Generated MIDI File
```

## Requirements

- Python 3.9+
- Streamlit
- Music Brain dependencies (see `requirements.txt`)

## Troubleshooting

### API Connection Issues

If you see "API unavailable" messages:
- Check if FastAPI service is running (if using API mode)
- Set `KMIDI_USE_API=false` to use direct API
- Check network/firewall settings

### Import Errors

If you see import errors:
- Install dependencies: `pip install -r requirements.txt`
- Ensure you're in the project root directory
- Check Python version: `python --version` (should be 3.9+)

### Generation Failures

If music generation fails:
- Check console for error messages
- Ensure music_brain dependencies are installed
- Try with simpler emotion text first
- Check available disk space for MIDI files

## Features Roadmap

- [ ] Real-time audio preview (Web Audio API)
- [ ] MIDI visualization (piano roll)
- [ ] Chord progression visualization
- [ ] Emotion timeline editor
- [ ] Batch generation
- [ ] User accounts and saved generations
- [ ] Social sharing

## License

See main project LICENSE file.

## Support

For issues or questions:
- Check the main project README
- Open an issue on GitHub
- Check documentation in `docs/` folder
