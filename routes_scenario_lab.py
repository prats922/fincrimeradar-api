"""
Scenario Lab route for the fincrimeradar-api service.

Read only, unauthenticated by design, per PRD section 9: this endpoint returns
no sensitive or production screening data, only static synthetic case fixtures.
Do not extend this pattern to expose real sanctions match results without
adding authentication first.

Confirmed against the actual service: main.py is FastAPI, not Flask, so this
ships as an APIRouter. Wire it into main.py with:

    from routes_scenario_lab import router as scenario_lab_router
    app.include_router(scenario_lab_router)

Rate limiting here is a small self contained in-memory sliding window, not a
new dependency. main.py currently has no rate limiting anywhere, adding a
library like slowapi is a reasonable follow up for /api/screen specifically,
but that is a decision for you to make given it touches requirements.txt on
a shared production service, not something to add silently as a side effect
of this one static fixture route.
"""

import json
import time
from collections import deque
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

CASES_PATH = Path(__file__).parent / "cases.json"
RATE_LIMIT_MAX_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60

router = APIRouter()

# Per client IP, a deque of request timestamps within the current window.
# In-memory, single-process only, resets on restart. That is an accepted
# limitation for a low-value static fixture endpoint on a single Render
# instance. If this service ever runs multiple instances behind a load
# balancer, this must move to a shared store (Redis) or the limit becomes
# per-instance rather than global, silently weaker than it looks.
_request_log = {}


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(request: Request) -> bool:
    now = time.monotonic()
    key = _client_key(request)
    log = _request_log.setdefault(key, deque())

    while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()

    if len(log) >= RATE_LIMIT_MAX_REQUESTS:
        return True

    log.append(now)
    return False


def _load_cases():
    if not CASES_PATH.exists():
        raise FileNotFoundError(
            f"cases.json not found at {CASES_PATH}. Confirm it was committed "
            "alongside this route file per PRD section 7."
        )
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Load once at import time. Phase 0 data is static and small, three cases,
# so there is no benefit to re-reading disk on every request, and doing so
# would be an unnecessary load vector on a shared Render service.
try:
    _CASES_CACHE = _load_cases()
    _LOAD_ERROR = None
except FileNotFoundError as exc:
    _CASES_CACHE = None
    _LOAD_ERROR = str(exc)


@router.get("/scenario-lab/cases")
def get_scenario_lab_cases(request: Request):
    if _rate_limited(request):
        raise HTTPException(status_code=429, detail="Too many requests. Try again shortly.")

    if _CASES_CACHE is None:
        # Fail loudly in logs, fail safely to the client. Do not leak the
        # filesystem path in the API response.
        print(f"Scenario Lab route error: {_LOAD_ERROR}")
        return JSONResponse(
            status_code=503,
            content={"error": "Scenario Lab case data is unavailable."},
        )

    # Bare array, matching the frontend's fetchCases contract exactly,
    # confirmed against scenario-lab.js: it expects Array.isArray(data)
    # directly, not a {"cases": [...]} wrapper.
    return JSONResponse(content=_CASES_CACHE)
