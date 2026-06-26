from fastapi import FastAPI, Query, HTTPException
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, json, re
from screening import SanctionsEngine, PEPEngine, AdverseMediaEngine

app = FastAPI(title="FinCrimeRadar API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fincrimeradar.org", "http://localhost:3000", "*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

sanctions_engine = SanctionsEngine()
pep_engine = PEPEngine()
adverse_engine = AdverseMediaEngine()

@app.on_event("startup")
async def startup():
    print("Loading screening data...")
    pep_engine.load()
    sanctions_engine.load()
    print("Screening engines ready.")
    asyncio.create_task(auto_refresh())

async def auto_refresh():
    """Background task — checks every hour if data needs refreshing."""
    while True:
        await asyncio.sleep(3600)  # check every hour
        try:
            from screening import cache_is_fresh
            sanctions_cache = "/tmp/sanctions_v3.json"
            pep_cache = "/tmp/peps_v4.json"
            if not cache_is_fresh(sanctions_cache):
                print("Auto-refresh: sanctions data stale, reloading...")
                sanctions_engine.load()
                print("Auto-refresh: sanctions updated")
            if not cache_is_fresh(pep_cache):
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
            "sanctions_cache_age": cache_age("/tmp/sanctions_lite.json"),
            "pep_records": pep_engine.count(),
            "pep_cache_age": cache_age("/tmp/peps_lite.json"),
            "cache_ttl_hours": 24,
            "adverse_media": "live RSS (no cache)",
        },
        "sources": [
            "OpenSanctions (OFAC, UN, EU, OFSI, 40+ lists)",
            "OpenSanctions PEP dataset",
            "BBC, OCCRP, AP, DW, Al Jazeera (RSS)",
        ]
    }
