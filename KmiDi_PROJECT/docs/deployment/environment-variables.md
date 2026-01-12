# Environment Variables for KmiDi Project

This document outlines the essential environment variables required to configure and run the KmiDi Music Generation API and related services.

## Core Application Settings

| Variable           | Description                                                                 | Default Value       | Example                               |
|--------------------|-----------------------------------------------------------------------------|---------------------|---------------------------------------|
| `KELLY_AUDIO_DATA_ROOT` | Root directory for audio data, models, and other assets. This should point to a persistent storage location. | `/app/data`         | `/usr/local/kmidi/data`               |

## API Service Settings (FastAPI)

| Variable           | Description                                                                 | Default Value       | Example                               |
|--------------------|-----------------------------------------------------------------------------|---------------------|---------------------------------------|
| `API_HOST`         | Host for the FastAPI server. For local development, use `127.0.0.1`. For Docker/production, use `0.0.0.0`. | `127.0.0.1`         | `0.0.0.0` or `your.domain.com`        |
| `API_PORT`         | Port for the FastAPI server.                                                | `8000`              | `8080`                                |
| `LOG_LEVEL`        | Logging level for the API service. Can be `INFO`, `DEBUG`, `WARNING`, `ERROR`, `CRITICAL`. | `INFO`              | `DEBUG`                               |
| `CORS_ORIGINS`     | Comma-separated list of allowed CORS origins for the web frontend.          | `http://localhost:1420` | `http://localhost:3000,https://app.yourdomain.com` |

## ML Model Settings (Optional)

| Variable           | Description                                                                 | Default Value       | Example                               |
|--------------------|-----------------------------------------------------------------------------|---------------------|---------------------------------------|
| `ENABLE_RTNEURAL`  | Enable/disable RTNeural for real-time inference. Set to `true` or `false`.   | `false`             | `true`                                |
| `ENABLE_ONNX_RUNTIME` | Enable/disable ONNX Runtime for ML inference. Set to `true` or `false`.   | `false`             | `true`                                |
| `ONNX_MODELS_DIR`  | Path to the directory containing ONNX models (if `ENABLE_ONNX_RUNTIME` is `true`). | `/app/models`       | `/usr/local/kmidi/models`             |

## Database Settings (if applicable)

| Variable           | Description                                                                 | Example                               |
|--------------------|-----------------------------------------------------------------------------|---------------------------------------|
| `DATABASE_URL`     | Connection URL for the database (e.g., PostgreSQL).                         | `postgresql://user:password@host:port/database_name` |
