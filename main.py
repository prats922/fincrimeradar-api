from fastapi import FastAPI, Query, HTTPException
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, json, re
from screening import SanctionsEngine, PEPEngine, AdverseMediaEngine, cache_is_fresh, SANCTIONS_CACHE_PATH, PEP_CACHE_PATH
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

sanctions_engine = SanctionsEngine()
pep_engine = PEPEngine()
adverse_engine = AdverseMediaEngine()

@app.on_event("startup")
async def startup():
    # No longer an unconditional wipe. The previous version cleared every
    # /tmp/*.json file on every restart before calling load(), which forced
    # a full re-download and re-parse of both datasets on every cold start
    # regardless of whether the existing cache was still genuinely fresh,
    # the exact cost a free-tier host spinning down on inactivity pays on
    # its very next request. load() itself already checks cache_is_fresh
    # internally and reads straight from cache when possible, so this step
    # now only removes a cache file when it has actually gone stale, the
    # same condition the hourly auto_refresh loop below already checks.
    print("Loading screening data...")
    if not cache_is_fresh(SANCTIONS_CACHE_PATH) and os.path.exists(SANCTIONS_CACHE_PATH):
        try:
            os.remove(SANCTIONS_CACHE_PATH)
            print(f"Cleared stale sanctions cache: {SANCTIONS_CACHE_PATH}")
        except OSError as e:
            print(f"Could not remove stale sanctions cache: {e}")
    if not cache_is_fresh(PEP_CACHE_PATH) and os.path.exists(PEP_CACHE_PATH):
        try:
            os.remove(PEP_CACHE_PATH)
            print(f"Cleared stale PEP cache: {PEP_CACHE_PATH}")
        except OSError as e:
            print(f"Could not remove stale PEP cache: {e}")

    pep_engine.load()
    sanctions_engine.load()
    print("Screening engines ready.")
    asyncio.create_task(auto_refresh())

async def auto_refresh():
    """Background task, checks every hour if data needs refreshing."""
    while True:
        await asyncio.sleep(3600)  # check every hour
        try:
            if not cache_is_fresh(SANCTIONS_CACHE_PATH):
                print("Auto-refresh: sanctions data stale, reloading...")
                sanctions_engine.load()
                print("Auto-refresh: sanctions updated")
            if not cache_is_fresh(PEP_CACHE_PATH):
                print("Auto-refresh: PEP data stale, reloading...")
                pep_engine.load()
                print("Auto-refresh: PEP updated")
        except Exception as e:
            print(f"Auto-refresh error: {e}")

@app.get("/")
def root():
    return {"service": "FinCrimeRadar API", "version": "1.0.0", "status": "live"}

@app.get("/api/screen")
def screen(
    q: str = Query(..., min_length=2, description="Name or entity to screen"),
    type: str = Query("all", description="all | sanctions | pep | adverse"),
    threshold: int = Query(80, ge=50, le=100, description="Match threshold 50-100")
):
    query = q.strip()
    results = {
        "query": query,
        "threshold": threshold,
        "sanctions": [],
        "pep": [],
        "adverse_media": [],
        "summary": {}
    }

    if type in ("all", "sanctions"):
        results["sanctions"] = sanctions_engine.search(query, threshold)

    if type in ("all", "pep"):
        results["pep"] = pep_engine.search(query, threshold)

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
        "sanctions_records": sanctions_engine.count(),
        "pep_records": pep_engine.count(),
    }

@app.get("/api/status")
def status():
    import os
    from datetime import datetime
    def cache_age(path):
        if not os.path.exists(path): return "not cached"
        modified = datetime.fromtimestamp(os.path.getmtime(path))
        age = datetime.now() - modified
        hours = age.seconds // 3600
        mins = (age.seconds % 3600) // 60
        return f"{hours}h {mins}m ago"

    return {
        "service": "FinCrimeRadar API",
        "version": "1.0.0",
        "status": "live",
        "data": {
            "sanctions_records": sanctions_engine.count(),
            "sanctions_cache_age": cache_age(SANCTIONS_CACHE_PATH),
            "sanctions_coverage": "Full consolidated sanctions target list (~70k entities)",
            "pep_records": pep_engine.count(),
            "pep_cache_age": cache_age(PEP_CACHE_PATH),
            "pep_coverage": "Partial: prioritised subset of heads of state, "
                             "senior government, and legislative roles. The full "
                             "OpenSanctions PEP dataset exceeds 750,000 entities "
                             "and is too large to host in full on this free tier. "
                             "Lower-profile regional/municipal PEPs may not be "
                             "covered. Always verify against official sources.",
            "cache_ttl_hours": 24,
            "adverse_media": "live RSS (no cache)",
        },
        "sources": [
            "OpenSanctions (OFAC, UN, EU, OFSI, 40+ lists)",
            "OpenSanctions PEP dataset (partial coverage, see pep_coverage)",
            "BBC, OCCRP, AP, DW, Al Jazeera (RSS)",
        ]
    }
