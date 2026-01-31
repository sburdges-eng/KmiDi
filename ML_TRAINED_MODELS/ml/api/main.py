from __future__ import annotations

from fastapi import FastAPI, Query
from pydantic import BaseModel

from ml.device_picker import DeviceInfo, select_device, suggest_settings


class DeviceResponse(BaseModel):
    backend: str
    device: str
    precision: str
    reason: str
    available: bool
    details: dict

    @classmethod
    def from_info(cls, info: DeviceInfo) -> "DeviceResponse":
        return cls(**info.to_dict())


class SuggestResponse(BaseModel):
    device: DeviceResponse
    suggested: dict


app = FastAPI(title="Device Picker", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/device", response_model=DeviceResponse)
def get_device(
    preference: str | None = Query(None, description="cuda|mps|mlx|cpu"),
) -> DeviceResponse:
    info = select_device(preference)
    return DeviceResponse.from_info(info)


@app.get("/suggest", response_model=SuggestResponse)
def get_suggestion(
    preference: str | None = Query(None, description="cuda|mps|mlx|cpu"),
) -> SuggestResponse:
    info = select_device(preference)
    return SuggestResponse(
        device=DeviceResponse.from_info(info),
        suggested=suggest_settings(info),
    )
