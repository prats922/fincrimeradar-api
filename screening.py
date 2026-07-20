import os, re, time, requests
from dotenv import load_dotenv

# main.py imports `screening` before it imports routes_guide_chat, which is
# the module that currently calls load_dotenv(). Reading OPENSANCTIONS_API_KEY
# at this module's import time would silently see an empty key locally if we
# relied on that later call. Calling it here too makes this module correct
# on its own regardless of import order elsewhere.
load_dotenv()

OPENSANCTIONS_API_KEY = os.environ.get("OPENSANCTIONS_API_KEY", "").strip()
OPENSANCTIONS_MATCH_URL = "https://api.opensanctions.org/match/default"
# Fixed, low, server-side threshold: cast a wide net once per screen, then
# filter locally by the user's slider value against the real scores already
# in hand. Confirmed 2026-07-20 this is cheaper and more correct than
# re-querying per threshold change, matches the existing UI behaviour where
# the slider only re-fetches when "Screen" is clicked, not on drag.
FETCH_THRESHOLD = 0.3
# Hard API ceiling, confirmed 2026-07-20 via real 422 responses: limit > 500
# is rejected outright ("Input should be less than or equal to 500"). This
# is a cap, not a safety margin, see the truncation warning in
# ScreeningEngine.match() for what happens when a name's real candidate
# pool exceeds it.
FETCH_LIMIT = 500

# GDELT_URL and PEP_WIKIDATA_URL previously declared here were already dead
# code before this rewrite (AdverseMediaEngine uses RSS_SOURCES, not GDELT;
# nothing ever read PEP_WIKIDATA_URL), left out rather than carried forward.

# Topics that put a result into the sanctions bucket vs the PEP bucket.
# Per OpenSanctions' own documented topic taxonomy: "sanction"/"sanction.linked"
# for designations, "role.pep"/"role.rca" for politically exposed persons and
# their relatives/close associates. An entity can carry both (e.g. Putin
# carries "sanction" and "role.pep" simultaneously in real API data checked
# 2026-07-20), and can therefore appear in both buckets, matching the old
# architecture where the same entity could independently exist in both the
# sanctions bulk file and the built-in PEP fallback list.
SANCTIONS_TOPICS = {"sanction", "sanction.linked"}
PEP_TOPICS = {"role.pep", "role.rca"}

# Short-lived in-memory cache for ScreeningEngine.match(), keyed on the
# normalized query alone, not (query, threshold): the underlying /match
# call always uses the fixed FETCH_THRESHOLD/FETCH_LIMIT above regardless
# of the user's slider value, threshold filtering happens downstream in
# ScreeningEngine.screen(). Keying on threshold too would cause a cache
# miss for the same name at two different slider positions even though
# the raw API response would be byte-identical, strictly worse for the
# actual goal here. TTL 1 hour: sanctions/PEP designations don't change
# on an hourly basis, and this directly cuts exposure to repeated
# searches of popular test names (Putin, Trump, etc.), which real
# visitor traffic will disproportionately consist of. Same in-memory,
# single-process, resets-on-restart tradeoff already accepted elsewhere
# in this codebase (routes_guide_chat.py, routes_scenario_lab.py), no
# explicit lock for the same reason those two don't use one either.
MATCH_CACHE_TTL_SECONDS = 60 * 60
_match_cache = {}


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip()).lower()


def dataset_label(ds: str) -> str:
    labels = {
        "us_ofac_sdn": "OFAC SDN (US)", "us_ofac_cons": "OFAC Consolidated (US)",
        "eu_fsf": "EU Financial Sanctions", "un_sc_sanctions": "UN Security Council",
        "gb_hmt_sanctions": "OFSI (UK)", "gb_fcdo_sanctions": "OFSI (UK)",
        "au_dfat_sanctions": "DFAT (Australia)",
        "ca_osfi_sanctions": "OSFI (Canada)", "ca_dfatd_sema_sanctions": "SEMA (Canada)",
        "ch_seco_sanctions": "SECO (Switzerland)",
    }
    return labels.get(ds, ds.upper().replace("_", " "))


# Generic/structural words that should not, on their own, drive a high match
# score. Used by AdverseMediaEngine below to keep RSS relevance matching from
# firing on common institutional terms alone (e.g. "bank", "management").
GENERIC_TERMS = {
    "the", "of", "for", "and", "an", "a", "in", "on", "at", "to",
    "bank", "banking", "institute", "institution", "management",
    "development", "state", "national", "international", "global",
    "group", "corporation", "corp", "company", "co", "ltd", "limited",
    "inc", "incorporated", "llc", "llp", "plc", "centre", "center",
    "foundation", "association", "agency", "authority", "council",
    "committee", "office", "ministry", "department", "bureau",
    "organization", "organisation", "society", "trust", "fund",
    "holdings", "holding", "enterprise", "enterprises", "industries",
    "industry", "services", "service", "solutions", "systems",
    "technologies", "technology", "university", "college", "school",
    "academy", "university", "general", "federal", "central", "union",
}


def _pep_categories(topics: list) -> list:
    """Best-effort category labels from topic substrings. Real OpenSanctions
    Person-level topics are only ever role.pep/role.rca in practice (finer
    classification like head-of-state lives on linked Position entities,
    not here), so this mostly yields "PEP". Kept anyway since it's harmless
    and matches the categorisation the old PEPEngine attempted."""
    cats = []
    for t in topics:
        if "head" in t: cats.append("Head of State")
        elif "gov" in t: cats.append("Government Official")
        elif "leg" in t: cats.append("Legislator")
        elif "diplo" in t: cats.append("Diplomat")
        elif "judge" in t: cats.append("Judiciary")
        elif "mil" in t: cats.append("Military")
        elif "soe" in t: cats.append("State Owned Enterprise")
        elif "pep" in t: cats.append("PEP")
    return list(set(cats)) or ["PEP"]


def _format_hit(result: dict, extra: dict) -> dict:
    props = result.get("properties", {})
    caption = result.get("caption", "")
    name_match = result.get("explanations", {}).get("name_match", {})
    matched_name = name_match.get("candidate") or caption
    hit = {
        "id": result.get("id"),
        "score": round(result.get("score", 0) * 100),
        "matched_name": matched_name,
        "primary_name": caption,
        "aliases": [n for n in props.get("name", []) if n != caption][:4],
        "entity_type": result.get("schema", "Unknown"),
        "nationality": props.get("nationality", []),
        "country": props.get("country", []),
        "birth_date": props.get("birthDate", []),
        "topics": props.get("topics", []),
    }
    hit.update(extra)
    return hit


def _build_sanctions_hit(result: dict) -> dict:
    props = result.get("properties", {})
    lists = [dataset_label(ds) for ds in result.get("datasets", [])]
    return _format_hit(result, {
        "sanctions_lists": lists or result.get("datasets", []),
        "program": props.get("programId", []),
        "position": props.get("position", []),
        # Closest real equivalent to the old bulk feed's "reason" field.
        # OpenSanctions doesn't expose a single dedicated reason property,
        # `notes` is the closest, a list of free-text description strings.
        "reason": props.get("notes", [])[:1],
    })


def _build_pep_hit(result: dict) -> dict:
    props = result.get("properties", {})
    return _format_hit(result, {
        "position": props.get("position", []),
        "pep_categories": _pep_categories(props.get("topics", [])),
    })


class ScreeningEngine:
    """Single authenticated OpenSanctions /match/default client, replacing
    the old SanctionsEngine/PEPEngine bulk-load-then-locally-fuzzy-match
    architecture entirely. One HTTP call per screen, two named queries
    (Person + LegalEntity) batched in one request body, confirmed
    2026-07-20 on the real billing dashboard to be metered as ONE request
    regardless of how many named queries are inside, not two. Sanctions
    and PEP results are split from this single response by topic after
    the fact (see SANCTIONS_TOPICS/PEP_TOPICS), not fetched separately.
    """

    def __init__(self):
        self.api_key = OPENSANCTIONS_API_KEY

    def _headers(self) -> dict:
        return {"Authorization": f"ApiKey {self.api_key}"}

    def match(self, query: str) -> list:
        """Returns a flat list of raw /match candidate dicts, deduplicated
        by id (highest score wins), filtered to target: true. Confirmed
        2026-07-20: both engines' predecessor data sources only ever
        contained genuine designated/PEP targets by construction (the
        sanctions-only and PEPs-only bulk collections), so this tool has
        never surfaced a pure related-party (target: false) result, and
        filtering here preserves that, not a new restriction."""
        cache_key = _normalize_query(query)
        cached = _match_cache.get(cache_key)
        if cached and (time.monotonic() - cached[0]) < MATCH_CACHE_TTL_SECONDS:
            print(f"Cache hit for {query!r}, skipping OpenSanctions /match call")
            return cached[1]

        if not self.api_key:
            print("OPENSANCTIONS_API_KEY not set, screening unavailable")
            return []

        body = {
            "queries": {
                "q_person": {"schema": "Person", "properties": {"name": [query]}},
                "q_org": {"schema": "LegalEntity", "properties": {"name": [query]}},
            }
        }
        params = {"threshold": FETCH_THRESHOLD, "limit": FETCH_LIMIT}
        try:
            resp = requests.post(
                OPENSANCTIONS_MATCH_URL, params=params, json=body,
                headers=self._headers(), timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"OpenSanctions /match call failed for {query!r}: {e}")
            return []

        combined = {}
        for qkey, qresp in data.get("responses", {}).items():
            results = qresp.get("results", [])
            total = qresp.get("total", {}).get("value", len(results))
            if len(results) < total:
                print(
                    f"WARNING: truncation for query '{qkey}' name={query!r}: "
                    f"received {len(results)} of {total} candidates "
                    f"(limit={FETCH_LIMIT}), real candidate pool exceeds the API cap"
                )
            for r in results:
                if not r.get("target"):
                    continue
                rid = r.get("id")
                if rid not in combined or r.get("score", 0) > combined[rid].get("score", 0):
                    combined[rid] = r
        result_list = list(combined.values())
        _match_cache[cache_key] = (time.monotonic(), result_list)
        return result_list

    def screen(self, query: str, threshold_pct: int):
        """One fetch, split into (sanctions, pep) lists by topic, each
        filtered to the caller's score threshold and capped at 8, matching
        the old engines' own result caps exactly."""
        raw = self.match(query)
        min_score = threshold_pct / 100.0
        sanctions, pep = [], []
        seen_sanctions, seen_pep = set(), set()
        for r in raw:
            if r.get("score", 0) < min_score:
                continue
            topics = set(r.get("properties", {}).get("topics", []) or [])
            rid = r.get("id")
            if topics & SANCTIONS_TOPICS and rid not in seen_sanctions:
                seen_sanctions.add(rid)
                sanctions.append(_build_sanctions_hit(r))
            if topics & PEP_TOPICS and rid not in seen_pep:
                seen_pep.add(rid)
                pep.append(_build_pep_hit(r))
        sanctions.sort(key=lambda x: x["score"], reverse=True)
        pep.sort(key=lambda x: x["score"], reverse=True)
        return sanctions[:8], pep[:8]


class AdverseMediaEngine:
    """
    Adverse media engine using RSS feeds from major news sources.
    No API key needed, no rate limits, always available.
    Sources: BBC, Reuters, Guardian, Al Jazeera, DW, AP News
    """

    RSS_SOURCES = [
        ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
        ("BBC Business", "http://feeds.bbci.co.uk/news/business/rss.xml"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("DW News", "https://rss.dw.com/rdf/rss-en-all"),
        ("AP News", "https://apnews.com/rss"),
        ("France 24", "https://www.france24.com/en/rss"),
        ("OCCRP", "https://www.occrp.org/en/rss"),
    ]

    def _parse_rss(self, url: str, timeout: int = 8) -> list:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; FinCrimeRadar/1.0)"}
            resp = requests.get(url, timeout=timeout, headers=headers)
            if resp.status_code != 200:
                return []

            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)

            items = []
            ns = {"media": "http://search.yahoo.com/mrss/"}

            for item in root.iter("item"):
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                pub_date = item.findtext("pubDate", "")[:16] if item.findtext("pubDate") else ""
                description = item.findtext("description", "").strip()
                items.append({
                    "title": title,
                    "url": link,
                    "description": description,
                    "date": pub_date,
                })
            return items
        except Exception as e:
            return []

    def _score_relevance(self, text: str, query_terms: list) -> int:
        text_lower = text.lower()
        score = 0
        meaningful_matches = 0
        for term in query_terms:
            if term.lower() in text_lower:
                score += 2
                meaningful_matches += 1
        # Require at least one genuinely distinguishing (non-generic) query
        # term to match before this article counts as relevant at all. Without
        # this, a query like "SDM Institute for Management" matches any
        # unrelated headline containing the word "institute" or "management",
        # since those are common institutional terms that say nothing about
        # the specific entity being screened.
        if meaningful_matches == 0:
            return 0
        # Bonus for financial crime keywords
        fincrime_terms = ["sanction", "fraud", "corrupt", "launder", "bribe", "crime",
                         "arrested", "charged", "convicted", "investigation", "penalty",
                         "fine", "regulatory", "banned", "blacklist", "illicit"]
        for term in fincrime_terms:
            if term in text_lower:
                score += 1
        return score

    def search(self, query: str) -> list:
        # Strip generic/structural words before matching, so a query like
        # "SDM Institute for Management" only triggers a hit on the
        # distinguishing word "SDM", not on generic institutional terms
        # like "institute" or "management" that appear in unrelated articles.
        query_terms = [t for t in query.strip().split() if len(t) > 2 and t.lower() not in GENERIC_TERMS]
        all_results = []

        for source_name, rss_url in self.RSS_SOURCES:
            try:
                items = self._parse_rss(rss_url)
                for item in items:
                    combined = f"{item['title']} {item['description']}"
                    score = self._score_relevance(combined, query_terms)
                    if score >= 2:  # at least name match + one fincrime term
                        all_results.append({
                            "title": item["title"],
                            "url": item["url"],
                            "source": source_name,
                            "date": item["date"][:10] if item["date"] else "",
                            "language": "English",
                            "tone": -3.0,  # negative by default for fincrime hits
                            "tone_label": "Negative",
                            "relevance_score": score,
                        })
            except Exception as e:
                print(f"RSS {source_name} failed: {e}")
                continue

        # Sort by relevance score then deduplicate by source
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        seen_sources = set()
        final = []
        for r in all_results:
            if r["source"] not in seen_sources:
                seen_sources.add(r["source"])
                final.append(r)
            if len(final) >= 8:
                break

        print(f"Adverse media RSS: found {len(final)} relevant articles for '{query}'")
        return final
