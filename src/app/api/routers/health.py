from fastapi import APIRouter

health_router = APIRouter()

@health_router.get("/health")

async def health() -> dict:
    """Simple health check for readiness/liveness probes."""
    return {"status": "ok"}