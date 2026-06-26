import json, os, re, requests
from rapidfuzz import fuzz, process
from datetime import datetime

OPENSANCTIONS_URL = "https://data.opensanctions.org/datasets/latest/sanctions/entities.ftm.json"
PEP_URL = "https://data.opensanctions.org/datasets/latest/peps/entities.ftm.json"

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def clean(name: str) -> str:
    return re.sub(r'\s+', ' ', name.strip().upper())

def safe_get(obj, *keys, default=""):
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        elif isinstance(obj, list):
            return obj
        else:
            return default
    return obj if obj else default

def extract_names(props: dict) -> list:
    names = []
    for field in ["name", "alias", "weakAlias"]:
        val = props.get(field, [])
        if isinstance(val, list):
            names.extend([v for v in val if v])
        elif val:
            names.append(val)
    return list(set(names))


class SanctionsEngine:
    def __init__(self):
        self.records = []
        self.name_index = []

    def load(self):
        cache = "/tmp/sanctions.json"
        if os.path.exists(cache):
            print("Loading sanctions from cache...")
            with open(cache) as f:
                self.records = json.load(f)
        else:
            print("Downloading sanctions data from OpenSanctions...")
            try:
                resp = requests.get(OPENSANCTIONS_URL, stream=True, timeout=60)
                records = []
                for line in resp.iter_lines():
                    if not line: continue
                    try:
                        entity = json.loads(line)
                        props = entity.get("properties", {})
                        names = extract_names(props)
                        if not names: continue
                        records.append({
                            "id": entity.get("id", ""),
                            "schema": entity.get("schema", ""),
                            "names": names,
                            "primary_name": names[0] if names else "",
                            "datasets": entity.get("datasets", []),
                            "nationality": props.get("nationality", []),
                            "country": props.get("country", []),
                            "birthDate": props.get("birthDate", []),
                            "position": props.get("position", []),
                            "topics": entity.get("topics", []),
                            "program": props.get("program", []),
                            "reason": props.get("reason", []),
                            "sourceUrl": props.get("sourceUrl", []),
                        })
                    except: continue
                self.records = records
                with open(cache, "w") as f:
                    json.dump(records, f)
                print(f"Loaded {len(records)} sanctions records")
            except Exception as e:
                print(f"Failed to load sanctions: {e}")
                self.records = self._fallback_records()

        self.name_index = [(clean(n), i, n) for i, r in enumerate(self.records) for n in r["names"]]
        print(f"Sanctions index built: {len(self.name_index)} name entries")

    def _fallback_records(self):
        return [
            {"id": "fallback-1", "schema": "Person", "names": ["VLADIMIR PUTIN"], "primary_name": "VLADIMIR PUTIN",
             "datasets": ["us_ofac_sdn"], "nationality": ["RU"], "country": ["RU"], "birthDate": ["1952-10-07"],
             "position": ["President of Russia"], "topics": ["sanction"], "program": ["UKRAINE-EO13685"],
             "reason": ["Senior government official"], "sourceUrl": []},
            {"id": "fallback-2", "schema": "Organization", "names": ["WAGNER GROUP", "PMC WAGNER"],
             "primary_name": "WAGNER GROUP", "datasets": ["us_ofac_sdn", "eu_fsf"], "nationality": [],
             "country": ["RU"], "birthDate": [], "position": [], "topics": ["sanction"],
             "program": ["RUSSIA-EO14024"], "reason": ["Private military company"], "sourceUrl": []},
        ]

    def search(self, query: str, threshold: int = 80) -> list:
        q = clean(query)
        if not self.name_index:
            return []

        matches = process.extract(q, [n[0] for n in self.name_index], scorer=fuzz.WRatio, limit=20, score_cutoff=threshold)
        seen_ids = set()
        results = []
        for match_name, score, idx in matches:
            orig_idx = self.name_index[idx][1]
            record = self.records[orig_idx]
            rid = record["id"]
            if rid in seen_ids: continue
            seen_ids.add(rid)

            list_names = []
            for ds in record.get("datasets", []):
                label = self._dataset_label(ds)
                if label: list_names.append(label)

            results.append({
                "id": rid,
                "score": round(score),
                "matched_name": self.name_index[idx][2],
                "primary_name": record["primary_name"],
                "aliases": [n for n in record["names"] if n != record["primary_name"]][:5],
                "entity_type": record.get("schema", "Unknown"),
                "sanctions_lists": list_names if list_names else record.get("datasets", []),
                "program": record.get("program", []),
                "nationality": record.get("nationality", []),
                "country": record.get("country", []),
                "birth_date": record.get("birthDate", []),
                "position": record.get("position", []),
                "reason": record.get("reason", []),
                "topics": record.get("topics", []),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:10]

    def _dataset_label(self, ds: str) -> str:
        labels = {
            "us_ofac_sdn": "OFAC SDN (US)",
            "us_ofac_cons": "OFAC Consolidated (US)",
            "eu_fsf": "EU Financial Sanctions",
            "un_sc_sanctions": "UN Security Council",
            "gb_hmt_sanctions": "OFSI (UK)",
            "au_dfat_sanctions": "DFAT (Australia)",
            "ca_osfi_sanctions": "OSFI (Canada)",
            "ch_seco_sanctions": "SECO (Switzerland)",
            "fr_gels_avoirs": "French Sanctions",
            "ru_acf_sanctions": "Russia ACF",
        }
        return labels.get(ds, ds.upper().replace("_", " "))

    def count(self): return len(self.records)


class PEPEngine:
    def __init__(self):
        self.records = []
        self.name_index = []

    def load(self):
        cache = "/tmp/peps.json"
        if os.path.exists(cache):
            print("Loading PEP data from cache...")
            with open(cache) as f:
                self.records = json.load(f)
        else:
            print("Downloading PEP data from OpenSanctions...")
            try:
                resp = requests.get(PEP_URL, stream=True, timeout=60)
                records = []
                for line in resp.iter_lines():
                    if not line: continue
                    try:
                        entity = json.loads(line)
                        if entity.get("schema") not in ("Person", "Organization"): continue
                        props = entity.get("properties", {})
                        names = extract_names(props)
                        if not names: continue
                        records.append({
                            "id": entity.get("id", ""),
                            "schema": entity.get("schema", ""),
                            "names": names,
                            "primary_name": names[0] if names else "",
                            "position": props.get("position", []),
                            "nationality": props.get("nationality", []),
                            "country": props.get("country", []),
                            "birthDate": props.get("birthDate", []),
                            "topics": entity.get("topics", []),
                            "datasets": entity.get("datasets", []),
                        })
                    except: continue
                self.records = records[:50000]
                with open(cache, "w") as f:
                    json.dump(self.records, f)
                print(f"Loaded {len(self.records)} PEP records")
            except Exception as e:
                print(f"Failed to load PEP: {e}")
                self.records = self._fallback_records()

        self.name_index = [(clean(n), i, n) for i, r in enumerate(self.records) for n in r["names"]]
        print(f"PEP index built: {len(self.name_index)} name entries")

    def _fallback_records(self):
        return [
            {"id": "pep-fallback-1", "schema": "Person", "names": ["RISHI SUNAK"],
             "primary_name": "RISHI SUNAK", "position": ["Prime Minister of the United Kingdom"],
             "nationality": ["GB"], "country": ["GB"], "birthDate": ["1980-05-12"],
             "topics": ["role.pep"], "datasets": ["gb_coh_psc"]},
        ]

    def search(self, query: str, threshold: int = 80) -> list:
        q = clean(query)
        if not self.name_index: return []

        matches = process.extract(q, [n[0] for n in self.name_index], scorer=fuzz.WRatio, limit=20, score_cutoff=threshold)
        seen_ids = set()
        results = []
        for match_name, score, idx in matches:
            orig_idx = self.name_index[idx][1]
            record = self.records[orig_idx]
            rid = record["id"]
            if rid in seen_ids: continue
            seen_ids.add(rid)

            pep_categories = []
            for topic in record.get("topics", []):
                if "head" in topic: pep_categories.append("Head of State")
                elif "gov" in topic: pep_categories.append("Government Official")
                elif "role.pep" in topic: pep_categories.append("PEP")
                elif "leg" in topic: pep_categories.append("Legislator")
                elif "diplo" in topic: pep_categories.append("Diplomat")
                elif "judge" in topic: pep_categories.append("Judiciary")
                elif "mil" in topic: pep_categories.append("Military")
                elif "soe" in topic: pep_categories.append("State Owned Enterprise")

            results.append({
                "id": rid,
                "score": round(score),
                "matched_name": self.name_index[idx][2],
                "primary_name": record["primary_name"],
                "aliases": [n for n in record["names"] if n != record["primary_name"]][:5],
                "entity_type": record.get("schema", "Person"),
                "position": record.get("position", []),
                "nationality": record.get("nationality", []),
                "country": record.get("country", []),
                "birth_date": record.get("birthDate", []),
                "pep_categories": list(set(pep_categories)) if pep_categories else ["PEP"],
                "topics": record.get("topics", []),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:10]

    def count(self): return len(self.records)


class AdverseMediaEngine:
    def search(self, query: str) -> list:
        try:
            params = {
                "query": f'"{query}" financial crime OR fraud OR money laundering OR sanctions OR corruption OR bribery',
                "mode": "ArtList",
                "maxrecords": "10",
                "format": "json",
                "timespan": "12m",
                "sort": "DateDesc",
            }
            resp = requests.get(GDELT_URL, params=params, timeout=10)
            data = resp.json()
            articles = data.get("articles", [])
            results = []
            for a in articles[:8]:
                results.append({
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "source": a.get("domain", ""),
                    "date": a.get("seendate", "")[:10] if a.get("seendate") else "",
                    "language": a.get("language", ""),
                    "tone": round(float(a.get("tone", 0)), 2),
                })
            return results
        except Exception as e:
            print(f"Adverse media search failed: {e}")
            return []
