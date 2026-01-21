# Starting the Music Brain API

The frontend application requires the Music Brain API to be running in the background.

## Quick Start

Run this command in a terminal:

```bash
./start-api.sh
```

Or manually:

```bash
python3 -m music_brain.api
```

The API will start at: **http://127.0.0.1:8000**

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
