import time
from collections import deque
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from screening import ScreeningEngine, AdverseMediaEngine, OPENSANCTIONS_API_KEY
from routes_scenario_lab import router as scenario_lab_router
from routes_guide_chat import router as guide_chat_router

app = FastAPI(title="FinCrimeRadar API", version="1.0.0")
app.include_router(scenario_lab_router)
app.include_router(guide_chat_router)

app.add_middleware(
    CORSMiddleware,
    # Wildcard removed. A bare "*" here made the two named origins below
    # meaningless, Starlette's CORS middleware treats "*" as allow every
    # origin outright, which left /api/screen scrapable from any website
    # on the internet with zero rate limiting behind it. Only the two
    # origins that genuinely need to call this API stay listed. Same
    # allowlist covers /api/chat, no new origins added for it, POST added
    # to allow_methods since /api/chat needs it (every other route here is
    # GET).
    allow_origins=["https://fincrimeradar.org", "https://www.fincrimeradar.org", "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

screening_engine = ScreeningEngine()
adverse_engine = AdverseMediaEngine()

# Per-IP rate limiting for /api/screen, same deque-of-timestamps pattern and
# same x-forwarded-for-aware IP extraction already used by
# routes_scenario_lab.py's /scenario-lab/cases (mirrored here rather than
# routes_guide_chat.py's session_id-keyed version, since /api/screen has no
# session concept, same as scenario-lab's endpoint). Starting number, not a
# permanent decision: this now calls a real, paid, per-request OpenSanctions
# API with no free minimum (the CORS comment above already flagged this
# endpoint as "scrapable from any website on the internet with zero rate
# limiting behind it" before this fix), so the number is chosen to sit in
# the same conservative order of magnitude as the established /api/chat cap
# (20 messages/24h) rather than scenario-lab's looser 60/60s (that endpoint
# serves static local case data with no external per-call cost). Used as a
# per-hour window instead of per-day: a real screening session plausibly
# involves checking multiple names in one sitting (e.g. working through a
# customer list), which a daily-only cap could exhaust in a single session.
# Revisit once this endpoint has its own real usage data. Same in-memory,
# single-process, resets-on-restart tradeoff already accepted for
# /api/chat and /scenario-lab/cases.
SCREEN_RATE_LIMIT_MAX = 20
SCREEN_RATE_LIMIT_WINDOW_SECONDS = 60 * 60

_screen_request_log = {}


def _screen_client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _screen_rate_limited(request: Request) -> bool:
    now = time.monotonic()
    key = _screen_client_key(request)
    log = _screen_request_log.setdefault(key, deque())

    while log and now - log[0] > SCREEN_RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()

    if len(log) >= SCREEN_RATE_LIMIT_MAX:
        return True

    log.append(now)
    return False

@app.on_event("startup")
async def startup():
    # No bulk dataset to load anymore, sanctions and PEP screening now call
    # OpenSanctions' authenticated /match/default per request (see
    # screening.py's ScreeningEngine), replacing the old bulk-download-and-
    # cache architecture entirely. The only startup-relevant check left is
    # whether the API key is actually configured, since a missing key would
    # otherwise fail silently on the first real search instead of being
    # visible in the startup log.
    if not OPENSANCTIONS_API_KEY:
        print("WARNING: OPENSANCTIONS_API_KEY is not set, sanctions/PEP screening will return no results")
    else:
        print("OpenSanctions API key configured, screening ready.")

@app.get("/")
def root():
    return {"service": "FinCrimeRadar API", "version": "1.0.0", "status": "live"}

@app.get("/api/screen")
def screen(
    request: Request,
    q: str = Query(..., min_length=2, description="Name or entity to screen"),
    type: str = Query("all", description="all | sanctions | pep | adverse"),
    threshold: int = Query(80, ge=50, le=100, description="Match threshold 50-100")
):
    if _screen_rate_limited(request):
        raise HTTPException(
            status_code=429,
            detail=f"Screening limit reached ({SCREEN_RATE_LIMIT_MAX} per hour). Try again later.",
        )

    query = q.strip()
    results = {
        "query": query,
        "threshold": threshold,
        "sanctions": [],
        "pep": [],
        "adverse_media": [],
        "summary": {}
    }

    if type in ("all", "sanctions", "pep"):
        # One call serves both, confirmed 2026-07-20 that a two-named-query
        # /match/default request bills as a single request regardless of
        # `type`, so this always fetches both even when only one is
        # requested, rather than risking two billed calls if this were ever
        # split into two conditional fetches.
        sanctions, pep = screening_engine.screen(query, threshold)
        if type in ("all", "sanctions"):
            results["sanctions"] = sanctions
        if type in ("all", "pep"):
            results["pep"] = pep

    if type in ("all", "adverse"):
        results["adverse_media"] = adverse_engine.search(query)

    total_hits = len(results["sanctions"]) + len(results["pep"]) + len(results["adverse_media"])
    max_score = 0
    if results["sanctions"]: max_score = max(max_score, results["sanctions"][0]["score"])
    if results["pep"]: max_score = max(max_score, results["pep"][0]["score"])

    risk = "clear"
    if total_hits > 0:
        if max_score >= 95 or len(results["sanctions"]) > 0:
            risk = "high"
        elif max_score >= 85 or len(results["pep"]) > 0:
            risk = "medium"
        else:
            risk = "low"

    results["summary"] = {
        "total_hits": total_hits,
        "sanctions_hits": len(results["sanctions"]),
        "pep_hits": len(results["pep"]),
        "adverse_hits": len(results["adverse_media"]),
        "risk_level": risk,
        "highest_score": max_score
    }

    return JSONResponse(content=results)

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "screening_backend": "opensanctions_api",
        "api_key_configured": bool(OPENSANCTIONS_API_KEY),
    }

@app.get("/api/status")
def status():
    return {
        "service": "FinCrimeRadar API",
        "version": "1.0.0",
        "status": "live",
        "data": {
            "screening_backend": "OpenSanctions authenticated /match/default API, "
                                  "live per-request, no bulk dataset held locally, "
                                  "no cache to go stale.",
            "api_key_configured": bool(OPENSANCTIONS_API_KEY),
            "sanctions_coverage": "Full OpenSanctions default collection, sanctions "
                                   "and PEP, split by topic per result. See methodology.html.",
            "adverse_media": "live RSS (no cache)",
        },
        "sources": [
            "OpenSanctions (OFAC, UN, EU, OFSI, 40+ lists, and PEP data), live API",
            "BBC, OCCRP, AP, DW, Al Jazeera (RSS)",
        ]
    }
