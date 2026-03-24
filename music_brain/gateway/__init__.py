"""
KMiDi Gateway API — single FastAPI service serving all three product tiers.

Tier 1 (Consumer): POST /api/v1/generate
Tier 2 (Pro):      WebSocket /api/v1/ws/plugin
Tier 3 (Premium):  WebSocket /api/v1/ws/daiw

Start with:
    uvicorn music_brain.gateway.app:app --port 8001
"""

__all__ = ["app"]
