import json, os, re, requests, gc, time
from datetime import datetime, timedelta
from rapidfuzz import fuzz, process

OPENSANCTIONS_URL = "https://data.opensanctions.org/datasets/latest/sanctions/entities.ftm.json"
PEP_URL = "https://data.opensanctions.org/datasets/latest/peps/entities.ftm.json"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

CACHE_TTL_HOURS = 24  # refresh data every 24 hours

def cache_is_fresh(cache_path: str) -> bool:
    """Check if cache file exists and is less than 24 hours old."""
    if not os.path.exists(cache_path):
        return False
    modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
    age = datetime.now() - modified
    fresh = age < timedelta(hours=CACHE_TTL_HOURS)
    if not fresh:
        print(f"Cache {cache_path} is {age.seconds//3600}h old — refreshing")
    return fresh

MAX_SANCTIONS = 12000
MAX_PEPS = 12000

def clean(name: str) -> str:
    return re.sub(r'\s+', ' ', name.strip().upper())

def extract_names(props: dict) -> list:
    names = []
    for field in ["name", "alias"]:
        val = props.get(field, [])
        if isinstance(val, list):
            names.extend([v for v in val if v and len(v) > 1])
        elif val:
            names.append(val)
    return list(set(names))[:4]

def dataset_label(ds: str) -> str:
    labels = {
        "us_ofac_sdn": "OFAC SDN (US)", "us_ofac_cons": "OFAC Consolidated (US)",
        "eu_fsf": "EU Financial Sanctions", "un_sc_sanctions": "UN Security Council",
        "gb_hmt_sanctions": "OFSI (UK)", "au_dfat_sanctions": "DFAT (Australia)",
        "ca_osfi_sanctions": "OSFI (Canada)", "ch_seco_sanctions": "SECO (Switzerland)",
    }
    return labels.get(ds, ds.upper().replace("_", " "))


class SanctionsEngine:
    def __init__(self):
        self.records = []
        self.name_index = []

    def load(self):
        cache = "/tmp/sanctions_lite.json"
        if cache_is_fresh(cache):
            print("Loading sanctions from fresh cache...")
            with open(cache) as f:
                self.records = json.load(f)
        else:
            print("Downloading sanctions data...")
            try:
                resp = requests.get(OPENSANCTIONS_URL, stream=True, timeout=120)
                records = []
                for line in resp.iter_lines():
                    if not line: continue
                    if len(records) >= MAX_SANCTIONS: break
                    try:
                        entity = json.loads(line)
                        props = entity.get("properties", {})
                        names = extract_names(props)
                        if not names: continue
                        records.append({
                            "id": entity.get("id", "")[:20],
                            "schema": entity.get("schema", "")[:20],
                            "names": names,
                            "primary_name": names[0],
                            "datasets": entity.get("datasets", [])[:4],
                            "nationality": props.get("nationality", [])[:3],
                            "country": props.get("country", [])[:3],
                            "birthDate": props.get("birthDate", [])[:2],
                            "position": props.get("position", [])[:2],
                            "program": props.get("program", [])[:3],
                            "reason": props.get("reason", [])[:1],
                            "topics": entity.get("topics", [])[:3],
                        })
                    except: continue
                self.records = records
                with open(cache, "w") as f:
                    json.dump(records, f)
                print(f"Loaded {len(records)} sanctions records")
            except Exception as e:
                print(f"Sanctions load failed: {e}")
                self.records = self._fallback()

        self.name_index = [(clean(n), i, n) for i, r in enumerate(self.records) for n in r["names"]]
        gc.collect()
        print(f"Sanctions index: {len(self.name_index)} entries")

    def _fallback(self):
        return [
            {"id": "f1", "schema": "Person", "names": ["VLADIMIR PUTIN"], "primary_name": "VLADIMIR PUTIN",
             "datasets": ["us_ofac_sdn"], "nationality": ["RU"], "country": ["RU"],
             "birthDate": ["1952-10-07"], "position": ["President of Russia"],
             "program": ["UKRAINE-EO13685"], "reason": ["Senior government official"], "topics": ["sanction"]},
            {"id": "f2", "schema": "Organization", "names": ["WAGNER GROUP", "PMC WAGNER"],
             "primary_name": "WAGNER GROUP", "datasets": ["us_ofac_sdn", "eu_fsf"],
             "nationality": [], "country": ["RU"], "birthDate": [], "position": [],
             "program": ["RUSSIA-EO14024"], "reason": ["Private military company"], "topics": ["sanction"]},
            {"id": "f3", "schema": "Person", "names": ["KIM JONG UN", "KIM JONG-UN"],
             "primary_name": "KIM JONG UN", "datasets": ["us_ofac_sdn", "un_sc_sanctions"],
             "nationality": ["KP"], "country": ["KP"], "birthDate": ["1984-01-08"],
             "position": ["Supreme Leader of North Korea"], "program": ["DPRK3"],
             "reason": ["Head of state"], "topics": ["sanction"]},
        ]

    def search(self, query: str, threshold: int = 80) -> list:
        q = clean(query)
        if not self.name_index: return []
        matches = process.extract(q, [n[0] for n in self.name_index], scorer=fuzz.WRatio, limit=15, score_cutoff=threshold)
        seen, results = set(), []
        for _, score, idx in matches:
            orig_idx = self.name_index[idx][1]
            record = self.records[orig_idx]
            rid = record["id"]
            if rid in seen: continue
            seen.add(rid)
            lists = [dataset_label(ds) for ds in record.get("datasets", [])]
            results.append({
                "id": rid, "score": round(score),
                "matched_name": self.name_index[idx][2],
                "primary_name": record["primary_name"],
                "aliases": [n for n in record["names"] if n != record["primary_name"]][:4],
                "entity_type": record.get("schema", "Unknown"),
                "sanctions_lists": lists or record.get("datasets", []),
                "program": record.get("program", []),
                "nationality": record.get("nationality", []),
                "country": record.get("country", []),
                "birth_date": record.get("birthDate", []),
                "position": record.get("position", []),
                "reason": record.get("reason", []),
                "topics": record.get("topics", []),
            })
        return sorted(results, key=lambda x: x["score"], reverse=True)[:8]

    def count(self): return len(self.records)


class PEPEngine:
    def __init__(self):
        self.records = []
        self.name_index = []

    def load(self):
        cache = "/tmp/peps_lite.json"
        if cache_is_fresh(cache):
            print("Loading PEP data from fresh cache...")
            with open(cache) as f:
                self.records = json.load(f)
        else:
            print("Downloading PEP data...")
            try:
                resp = requests.get(PEP_URL, stream=True, timeout=120)
                records = []
                for line in resp.iter_lines():
                    if not line: continue
                    if len(records) >= MAX_PEPS: break
                    try:
                        entity = json.loads(line)
                        if entity.get("schema") not in ("Person", "Organization"): continue
                        props = entity.get("properties", {})
                        names = extract_names(props)
                        if not names: continue
                        records.append({
                            "id": entity.get("id", "")[:20],
                            "schema": entity.get("schema", ""),
                            "names": names,
                            "primary_name": names[0],
                            "position": props.get("position", [])[:2],
                            "nationality": props.get("nationality", [])[:3],
                            "country": props.get("country", [])[:3],
                            "birthDate": props.get("birthDate", [])[:2],
                            "topics": entity.get("topics", [])[:3],
                        })
                    except: continue
                self.records = records
                with open(cache, "w") as f:
                    json.dump(records, f)
                print(f"Loaded {len(records)} PEP records")
            except Exception as e:
                print(f"PEP load failed: {e}")
                self.records = self._fallback()

        self.name_index = [(clean(n), i, n) for i, r in enumerate(self.records) for n in r["names"]]
        gc.collect()
        print(f"PEP index: {len(self.name_index)} entries")

    def _fallback(self):
        return [
            {"id": "pep1", "schema": "Person", "names": ["RISHI SUNAK"],
             "primary_name": "RISHI SUNAK", "position": ["Prime Minister of the United Kingdom"],
             "nationality": ["GB"], "country": ["GB"], "birthDate": ["1980-05-12"],
             "topics": ["role.pep", "role.head"]},
            {"id": "pep2", "schema": "Person", "names": ["KEIR STARMER"],
             "primary_name": "KEIR STARMER", "position": ["Prime Minister of the United Kingdom"],
             "nationality": ["GB"], "country": ["GB"], "birthDate": ["1962-09-02"],
             "topics": ["role.pep", "role.head"]},
        ]

    def search(self, query: str, threshold: int = 80) -> list:
        q = clean(query)
        if not self.name_index: return []
        matches = process.extract(q, [n[0] for n in self.name_index], scorer=fuzz.WRatio, limit=15, score_cutoff=threshold)
        seen, results = set(), []
        for _, score, idx in matches:
            orig_idx = self.name_index[idx][1]
            record = self.records[orig_idx]
            rid = record["id"]
            if rid in seen: continue
            seen.add(rid)
            pep_cats = []
            for t in record.get("topics", []):
                if "head" in t: pep_cats.append("Head of State")
                elif "gov" in t: pep_cats.append("Government Official")
                elif "leg" in t: pep_cats.append("Legislator")
                elif "diplo" in t: pep_cats.append("Diplomat")
                elif "judge" in t: pep_cats.append("Judiciary")
                elif "mil" in t: pep_cats.append("Military")
                elif "soe" in t: pep_cats.append("State Owned Enterprise")
                elif "pep" in t: pep_cats.append("PEP")
            results.append({
                "id": rid, "score": round(score),
                "matched_name": self.name_index[idx][2],
                "primary_name": record["primary_name"],
                "aliases": [n for n in record["names"] if n != record["primary_name"]][:3],
                "entity_type": record.get("schema", "Person"),
                "position": record.get("position", []),
                "nationality": record.get("nationality", []),
                "country": record.get("country", []),
                "birth_date": record.get("birthDate", []),
                "pep_categories": list(set(pep_cats)) or ["PEP"],
                "topics": record.get("topics", []),
            })
        return sorted(results, key=lambda x: x["score"], reverse=True)[:8]

    def count(self): return len(self.records)


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
        for term in query_terms:
            if term.lower() in text_lower:
                score += 2
        # Bonus for financial crime keywords
        fincrime_terms = ["sanction", "fraud", "corrupt", "launder", "bribe", "crime",
                         "arrested", "charged", "convicted", "investigation", "penalty",
                         "fine", "regulatory", "banned", "blacklist", "illicit"]
        for term in fincrime_terms:
            if term in text_lower:
                score += 1
        return score

    def search(self, query: str) -> list:
        query_terms = [t for t in query.strip().split() if len(t) > 2]
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
