# Starting the Music Brain API

The frontend application requires the Music Brain API to be running in the background.

## Quick Start

From the repo root:

```bash
python3 -m uvicorn music_brain.api:app --reload --port 8000 --host 0.0.0.0
```

Or with a custom host (e.g. local only):

```bash
python3 -m uvicorn music_brain.api:app --reload --port 8000
```

The API will start at: **http://127.0.0.1:8000** (see [AGENTS.md](../AGENTS.md) and `npm run dev:python`).

## If You Get Import Errors

If you see errors about missing packages, install dependencies:

```bash
pip3 install fastapi uvicorn
```

Or install all requirements:

```bash
pip3 install -r requirements.txt
```

## Keep It Running

- The API must stay running while you use the app
- Keep the terminal window open
- Press `Ctrl+C` to stop the server
- Restart it if you close the terminal

## Verify It's Running

Open in your browser: http://127.0.0.1:8000/health

You should see: `{"status":"ok","version":"0.1.0"}`
