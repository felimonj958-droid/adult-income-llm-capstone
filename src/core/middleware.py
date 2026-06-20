import time
from collections import defaultdict
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.logging import logger

RATE_LIMITS = {
    "/chat": {"limit": 5, "window_seconds": 60},
    "/explain-prediction": {"limit": 5, "window_seconds": 60},
}

request_history: dict[tuple[str, str], list[float]] = defaultdict(list)


async def rate_limit_requests(request: Request, call_next: Callable):
    path = request.url.path

    if path not in RATE_LIMITS:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    config = RATE_LIMITS[path]
    window_seconds = config["window_seconds"]
    limit = config["limit"]

    key = (client_ip, path)
    recent_requests = [
        timestamp for timestamp in request_history[key]
        if now - timestamp < window_seconds
    ]
    request_history[key] = recent_requests

    if len(recent_requests) >= limit:
        retry_after = max(1, int(window_seconds - (now - recent_requests[0])))
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."},
            headers={"Retry-After": str(retry_after)},
        )

    request_history[key].append(now)
    return await call_next(request)


async def log_requests(request: Request, call_next: Callable):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "%s %s -> %s in %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def register_middleware(app: FastAPI) -> None:
    app.middleware("http")(rate_limit_requests)
    app.middleware("http")(log_requests)
