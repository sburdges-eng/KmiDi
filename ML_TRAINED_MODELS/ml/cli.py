from __future__ import annotations

import json
from typing import Optional

import typer  # type: ignore[import-untyped]

from ml.device_picker import select_device

app = typer.Typer(help="Device-aware CLI for inference/training setup.")


@app.command()
def device(
    preference: Optional[str] = typer.Option(None, help="cuda|mps|mlx|cpu"),
) -> None:
    info = select_device(preference)
    typer.echo(json.dumps(info.to_dict(), indent=2))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host for the API"),
    port: int = typer.Option(8000, help="Bind port for the API"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
) -> None:
    import uvicorn  # type: ignore[import-untyped]

    uvicorn.run("ml.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
