import time
import uuid
from collections import defaultdict
from threading import Lock
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from src.core.logging import logger

RATE_LIMITS: dict[str, dict[str, int]] = {
    "/chat": {"limit": 5, "window_seconds": 60},
    "/explain-prediction": {"limit": 5, "window_seconds": 60},
}

request_history: dict[tuple[str, str], list[float]] = defaultdict(list)
request_history_lock = Lock()


def _get_client_identifier(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def _cleanup_expired_requests(
    timestamps: list[float],
    now: float,
    window_seconds: int,
) -> list[float]:
    return [timestamp for timestamp in timestamps if now - timestamp < window_seconds]


async def rate_limit_requests(request: Request, call_next: Callable) -> Response:
    path = request.url.path
    config = RATE_LIMITS.get(path)

    if config is None:
        return await call_next(request)

    client_id = _get_client_identifier(request)
    now = time.time()
    limit = config["limit"]
    window_seconds = config["window_seconds"]
    key = (client_id, path)

    with request_history_lock:
        recent_requests = _cleanup_expired_requests(
            request_history[key],
            now,
            window_seconds,
        )
        request_history[key] = recent_requests

        if len(recent_requests) >= limit:
            retry_after = max(1, int(window_seconds - (now - recent_requests[0])))
            remaining = 0

            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Window": str(window_seconds),
                },
            )

        request_history[key].append(now)
        remaining = max(0, limit - len(request_history[key]))

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Window"] = str(window_seconds)
    return response


async def log_requests(request: Request, call_next: Callable) -> Response:
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_id=%s method=%s path=%s client=%s status=500 duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            _get_client_identifier(request),
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id

    logger.info(
        "request_id=%s method=%s path=%s client=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        _get_client_identifier(request),
        response.status_code,
        duration_ms,
    )
    return response


def register_middleware(app: FastAPI) -> None:
    app.middleware("http")(log_requests)
    app.middleware("http")(rate_limit_requests)
