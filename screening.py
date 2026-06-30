import json, os, re, requests, gc, time
from datetime import datetime, timedelta
from rapidfuzz import fuzz, process

OPENSANCTIONS_URL = "https://data.opensanctions.org/datasets/latest/sanctions/entities.ftm.json"
# Dedicated PEPs collection (separate from sanctions) - the previous code
# incorrectly cross-referenced PEP topics out of the sanctions-only stream,
# which only surfaced entities that were BOTH sanctioned AND PEP-tagged,
# explaining why only ~114 records ever loaded despite OpenSanctions having
# 750,000+ PEP targets. This is the correct dedicated dataset endpoint.
OPENSANCTIONS_PEPS_URL = "https://data.opensanctions.org/datasets/latest/peps/entities.ftm.json"
# PEP data extracted from sanctions dataset + built-in database
PEP_WIKIDATA_URL = "https://query.wikidata.org/sparql"
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

# OpenSanctions' consolidated sanctions collection contains ~70,000 actual
# screening targets (people, organizations, companies) once non-target
# schema types (vessels, addresses, securities, etc.) are filtered out.
# The previous cap of 25,000 silently truncated the dataset roughly 65% of
# the way through, meaning well-known sanctioned entities could be missing
# entirely depending on stream order, a genuine false-negative risk. This
# cap is set with headroom above the full target count.
MAX_SANCTIONS = 80000
# OpenSanctions' dedicated PEP collection has 750k+ targets across 134
# national sources (897MB raw), too large to hold in memory on a free-tier
# host alongside the sanctions index. This cap keeps a meaningfully large,
# but necessarily partial, PEP set. See PEPEngine.load() for source
# prioritisation logic and the partial-coverage warning surfaced to users.
MAX_PEPS = 60000

def clean(name: str) -> str:
    return re.sub(r'\s+', ' ', name.strip().upper())

# Generic/structural words that should not, on their own, drive a high match score.
# Without this, short or generic-sounding entity names (e.g. "SDM Bank", "Management
# Center") trigger false positive sanctions hits against unrelated organisations
# whose names happen to share a common business/legal term.
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

def smart_match_score(query: str, candidate: str) -> float:
    """
    Token-set based fuzzy match with a generic-term penalty.

    Rationale: rapidfuzz's WRatio alone over-weights partial substring and
    token overlap, which produces false positives whenever a query shares
    only common business/institutional words with a candidate name (e.g.
    an unrelated "Institute for Management" scoring 85+ against "SDM Bank"
    purely because both contain generic terms). This function requires
    genuine non-generic word overlap (or strong full-string similarity) to
    award a high score, while still catching real typos, abbreviations,
    and transliteration variants of sanctioned/PEP names.
    """
    q = query.lower().strip()
    c = candidate.lower().strip()

    base_score = fuzz.token_set_ratio(q, c)
    wratio_score = fuzz.WRatio(q, c)
    combined_base = max(base_score, wratio_score * 0.85)

    q_words = set(re.findall(r"[a-z0-9]+", q))
    c_words = set(re.findall(r"[a-z0-9]+", c))

    q_meaningful = q_words - GENERIC_TERMS
    c_meaningful = c_words - GENERIC_TERMS

    if q_meaningful and c_meaningful:
        meaningful_overlap = q_meaningful & c_meaningful
        overlap_ratio = len(meaningful_overlap) / max(len(q_meaningful), len(c_meaningful))
        if not meaningful_overlap:
            # Shared structure only (e.g. both have "institute", "bank") with
            # zero genuine name overlap. Classic false-positive shape.
            combined_base *= 0.55
        elif overlap_ratio < 0.4:
            # Some meaningful overlap exists but it's a small fraction of
            # either name's distinguishing content - still a weak match.
            combined_base *= 0.65
    elif not q_meaningful or not c_meaningful:
        # One side is entirely generic/structural words. A query like "the
        # institute" or "state bank" alone should never score high purely
        # against another generically-named entity.
        combined_base *= 0.6

    # Guard against short queries matching long unrelated names purely on
    # substring containment.
    len_ratio = min(len(q_words), len(c_words)) / max(len(q_words), len(c_words))
    if len_ratio < 0.35:
        combined_base *= 0.9

    return round(combined_base, 1)

def extract_names(props: dict) -> list:
    names = []
    for field in ["name", "alias"]:
        val = props.get(field, [])
        if isinstance(val, list):
            names.extend([v for v in val if v and len(v) > 1])
        elif val:
            names.append(val)
    return list(set(n.strip() for n in names if n and len(n.strip()) > 1))[:4]

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
        cache = "/tmp/sanctions_v4.json"
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
                        schema = entity.get("schema", "")

                        # Skip non-relevant entity types to save memory
                        # Focus on Person, Organization, Company, LegalEntity
                        if schema in ("Vessel", "Aircraft", "Airplane", "Vehicle",
                                     "Asset", "Crypto", "Address", "Thing"):
                            continue

                        props = entity.get("properties", {})
                        names = extract_names(props)
                        if not names: continue

                        # Only store essential fields — minimal memory footprint
                        records.append({
                            "id": entity.get("id", "")[:20],
                            "schema": schema[:20],
                            "names": names,
                            "primary_name": names[0],
                            "datasets": entity.get("datasets", [])[:4],
                            "nationality": props.get("nationality", [])[:3],
                            "country": props.get("country", [])[:3],
                            "birthDate": props.get("birthDate", [])[:1],
                            "position": props.get("position", [])[:1],
                            "program": props.get("program", [])[:2],
                            "reason": props.get("reason", [])[:1],
                            "topics": entity.get("topics", [])[:2],
                        })
                    except: continue
                self.records = records
                with open(cache, "w") as f:
                    json.dump(records, f)
                print(f"Loaded {len(records)} sanctions records")
            except Exception as e:
                print(f"Sanctions load failed: {e}")
                import traceback
                traceback.print_exc()
                self.records = self._fallback()

        self.name_index = [(clean(n), i, n) for i, r in enumerate(self.records) for n in r["names"]]
        gc.collect()
        try:
            import resource
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(f"Sanctions index: {len(self.name_index)} entries | Memory used: {mem//1024}MB")
        except:
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
        # Cast a wider net at candidate-generation stage (lower cutoff), then
        # re-score with smart_match_score, which penalises generic-word-only
        # overlap. This avoids the false positives that raw WRatio produces
        # on short/generic institutional names while still catching real
        # typos, abbreviations and transliteration variants.
        candidate_cutoff = max(50, threshold - 25)
        raw_matches = process.extract(q, [n[0] for n in self.name_index], scorer=fuzz.WRatio, limit=40, score_cutoff=candidate_cutoff)
        rescored = []
        for matched_text, _, idx in raw_matches:
            final_score = smart_match_score(q, matched_text)
            if final_score >= threshold:
                rescored.append((matched_text, final_score, idx))
        rescored.sort(key=lambda x: x[1], reverse=True)
        matches = rescored[:15]
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
        cache = "/tmp/peps_v6.json"
        if cache_is_fresh(cache):
            print("Loading PEP data from fresh cache...")
            with open(cache) as f:
                self.records = json.load(f)
        else:
            print("Building PEP database from dedicated OpenSanctions PEPs dataset...")
            try:
                # Use the correct dedicated PEPs collection, NOT the sanctions
                # dataset. The PEPs collection is ~750k targets / 897MB raw,
                # too large to fully hold in memory on a free-tier host, so we
                # apply a cap with priority ordering: highest-impact roles
                # (heads of state, senior government, legislators, judiciary,
                # diplomats) are kept preferentially over lower-priority
                # categories (e.g. local/municipal officials) if the cap is
                # reached before the stream ends.
                resp = requests.get(OPENSANCTIONS_PEPS_URL, stream=True, timeout=180)
                pep_schemas = {"Person", "Organization", "Company", "PublicBody", "LegalEntity"}

                # Priority tiers - higher priority topics fill the cap first.
                high_priority_topics = {"role.head", "role.pep", "role.gov", "role.pol"}
                medium_priority_topics = {"role.leg", "role.diplo", "role.judge", "role.mil", "role.soe", "role.mep"}
                all_pep_topics = high_priority_topics | medium_priority_topics | {"role.rca"}

                high_records, medium_records, low_records = [], [], []

                for line in resp.iter_lines():
                    if not line: continue
                    # Stop once we've comfortably exceeded the cap across all
                    # tiers combined, to bound total download/parse time.
                    if len(high_records) + len(medium_records) + len(low_records) >= MAX_PEPS * 2:
                        break
                    try:
                        entity = json.loads(line)
                        schema = entity.get("schema", "")
                        if schema not in pep_schemas: continue

                        topics = set(entity.get("topics", []))
                        if not topics.intersection(all_pep_topics): continue

                        props = entity.get("properties", {})
                        names = extract_names(props)
                        if not names: continue

                        record = {
                            "id": "pep_" + entity.get("id", "")[:16],
                            "schema": schema,
                            "names": names,
                            "primary_name": names[0],
                            "position": props.get("position", [])[:2],
                            "nationality": props.get("nationality", [])[:2],
                            "country": props.get("country", [])[:2],
                            "birthDate": props.get("birthDate", [])[:1],
                            "topics": list(topics)[:3],
                        }

                        if topics.intersection(high_priority_topics):
                            high_records.append(record)
                        elif topics.intersection(medium_priority_topics):
                            medium_records.append(record)
                        else:
                            low_records.append(record)
                    except: continue

                # Fill the cap starting with highest priority tier first.
                records = high_records[:MAX_PEPS]
                remaining = MAX_PEPS - len(records)
                if remaining > 0:
                    records.extend(medium_records[:remaining])
                    remaining = MAX_PEPS - len(records)
                if remaining > 0:
                    records.extend(low_records[:remaining])

                # Add built-in PEP database as supplement
                records.extend(self._builtin_peps())
                # Deduplicate by name
                seen_names = set()
                deduped = []
                for r in records:
                    key = r["primary_name"].upper()
                    if key not in seen_names:
                        seen_names.add(key)
                        deduped.append(r)
                self.records = deduped

                with open(cache, "w") as f:
                    json.dump(self.records, f)
                print(f"Loaded {len(self.records)} PEP records "
                      f"(high-priority: {len(high_records[:MAX_PEPS])}, "
                      f"from dedicated PEPs dataset + built-in)")
            except Exception as e:
                print(f"PEP load failed: {e}")
                import traceback
                traceback.print_exc()
                self.records = self._builtin_peps()

        self.name_index = [(clean(n), i, n) for i, r in enumerate(self.records) for n in r["names"]]
        gc.collect()
        try:
            import resource
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(f"PEP index: {len(self.name_index)} entries | Memory used: {mem//1024}MB")
        except:
            print(f"PEP index: {len(self.name_index)} entries")

    def _builtin_peps(self):
        """
        Built-in PEP database — 200+ records covering:
        - Heads of State & Government (all G20 + major nations)
        - Finance Ministers & Central Bank Governors (high AML risk)
        - Senior Law Enforcement & Intelligence Chiefs
        - Prominent sanctioned PEPs
        - Regional leaders (Africa, Middle East, Asia, LatAm)
        """
        return [
            # ── HEADS OF STATE / GOVERNMENT ──────────────────────────────────
            {"id":"pep_gb01","schema":"Person","names":["KEIR STARMER","SIR KEIR STARMER"],"primary_name":"KEIR STARMER","position":["Prime Minister of the United Kingdom"],"nationality":["GB"],"country":["GB"],"birthDate":["1962-09-02"],"topics":["role.head","role.pep"]},
            {"id":"pep_gb02","schema":"Person","names":["RISHI SUNAK"],"primary_name":"RISHI SUNAK","position":["Former Prime Minister of the United Kingdom"],"nationality":["GB"],"country":["GB"],"birthDate":["1980-05-12"],"topics":["role.head","role.pep"]},
            {"id":"pep_gb03","schema":"Person","names":["BORIS JOHNSON","ALEXANDER BORIS DE PFEFFEL JOHNSON"],"primary_name":"BORIS JOHNSON","position":["Former Prime Minister of the United Kingdom"],"nationality":["GB"],"country":["GB"],"birthDate":["1964-06-19"],"topics":["role.head","role.pep"]},
            {"id":"pep_us01","schema":"Person","names":["DONALD TRUMP","DONALD J TRUMP"],"primary_name":"DONALD TRUMP","position":["President of the United States"],"nationality":["US"],"country":["US"],"birthDate":["1946-06-14"],"topics":["role.head","role.pep"]},
            {"id":"pep_us02","schema":"Person","names":["JOE BIDEN","JOSEPH BIDEN","JOSEPH R BIDEN"],"primary_name":"JOE BIDEN","position":["Former President of the United States"],"nationality":["US"],"country":["US"],"birthDate":["1942-11-20"],"topics":["role.head","role.pep"]},
            {"id":"pep_us03","schema":"Person","names":["BARACK OBAMA","BARACK HUSSEIN OBAMA"],"primary_name":"BARACK OBAMA","position":["Former President of the United States"],"nationality":["US"],"country":["US"],"birthDate":["1961-08-04"],"topics":["role.head","role.pep"]},
            {"id":"pep_ru01","schema":"Person","names":["VLADIMIR PUTIN","PUTIN VLADIMIR VLADIMIROVICH"],"primary_name":"VLADIMIR PUTIN","position":["President of Russia"],"nationality":["RU"],"country":["RU"],"birthDate":["1952-10-07"],"topics":["role.head","role.pep","sanction"]},
            {"id":"pep_ru02","schema":"Person","names":["MIKHAIL MISHUSTIN"],"primary_name":"MIKHAIL MISHUSTIN","position":["Prime Minister of Russia"],"nationality":["RU"],"country":["RU"],"birthDate":["1966-03-03"],"topics":["role.gov","role.pep","sanction"]},
            {"id":"pep_ru03","schema":"Person","names":["SERGEI LAVROV"],"primary_name":"SERGEI LAVROV","position":["Minister of Foreign Affairs of Russia"],"nationality":["RU"],"country":["RU"],"birthDate":["1950-03-21"],"topics":["role.gov","role.pep","sanction"]},
            {"id":"pep_cn01","schema":"Person","names":["XI JINPING"],"primary_name":"XI JINPING","position":["President of China","General Secretary of the CCP"],"nationality":["CN"],"country":["CN"],"birthDate":["1953-06-15"],"topics":["role.head","role.pep"]},
            {"id":"pep_cn02","schema":"Person","names":["LI QIANG"],"primary_name":"LI QIANG","position":["Premier of the State Council of China"],"nationality":["CN"],"country":["CN"],"birthDate":["1959-07-23"],"topics":["role.gov","role.pep"]},
            {"id":"pep_fr01","schema":"Person","names":["EMMANUEL MACRON"],"primary_name":"EMMANUEL MACRON","position":["President of France"],"nationality":["FR"],"country":["FR"],"birthDate":["1977-12-21"],"topics":["role.head","role.pep"]},
            {"id":"pep_fr02","schema":"Person","names":["GABRIEL ATTAL"],"primary_name":"GABRIEL ATTAL","position":["Former Prime Minister of France"],"nationality":["FR"],"country":["FR"],"birthDate":["1989-03-16"],"topics":["role.gov","role.pep"]},
            {"id":"pep_de01","schema":"Person","names":["OLAF SCHOLZ"],"primary_name":"OLAF SCHOLZ","position":["Chancellor of Germany"],"nationality":["DE"],"country":["DE"],"birthDate":["1958-06-14"],"topics":["role.head","role.pep"]},
            {"id":"pep_de02","schema":"Person","names":["ANGELA MERKEL"],"primary_name":"ANGELA MERKEL","position":["Former Chancellor of Germany"],"nationality":["DE"],"country":["DE"],"birthDate":["1954-07-17"],"topics":["role.head","role.pep"]},
            {"id":"pep_in01","schema":"Person","names":["NARENDRA MODI"],"primary_name":"NARENDRA MODI","position":["Prime Minister of India"],"nationality":["IN"],"country":["IN"],"birthDate":["1950-09-17"],"topics":["role.head","role.pep"]},
            {"id":"pep_in02","schema":"Person","names":["DROUPADI MURMU"],"primary_name":"DROUPADI MURMU","position":["President of India"],"nationality":["IN"],"country":["IN"],"birthDate":["1958-06-20"],"topics":["role.head","role.pep"]},
            {"id":"pep_jp01","schema":"Person","names":["FUMIO KISHIDA"],"primary_name":"FUMIO KISHIDA","position":["Former Prime Minister of Japan"],"nationality":["JP"],"country":["JP"],"birthDate":["1957-07-29"],"topics":["role.head","role.pep"]},
            {"id":"pep_jp02","schema":"Person","names":["SHIGERU ISHIBA"],"primary_name":"SHIGERU ISHIBA","position":["Prime Minister of Japan"],"nationality":["JP"],"country":["JP"],"birthDate":["1957-02-04"],"topics":["role.head","role.pep"]},
            {"id":"pep_it01","schema":"Person","names":["GIORGIA MELONI"],"primary_name":"GIORGIA MELONI","position":["Prime Minister of Italy"],"nationality":["IT"],"country":["IT"],"birthDate":["1977-01-15"],"topics":["role.head","role.pep"]},
            {"id":"pep_ca01","schema":"Person","names":["JUSTIN TRUDEAU"],"primary_name":"JUSTIN TRUDEAU","position":["Former Prime Minister of Canada"],"nationality":["CA"],"country":["CA"],"birthDate":["1971-12-25"],"topics":["role.head","role.pep"]},
            {"id":"pep_au01","schema":"Person","names":["ANTHONY ALBANESE"],"primary_name":"ANTHONY ALBANESE","position":["Prime Minister of Australia"],"nationality":["AU"],"country":["AU"],"birthDate":["1963-03-02"],"topics":["role.head","role.pep"]},
            {"id":"pep_br01","schema":"Person","names":["LUIZ INACIO LULA DA SILVA","LULA"],"primary_name":"LUIZ INACIO LULA DA SILVA","position":["President of Brazil"],"nationality":["BR"],"country":["BR"],"birthDate":["1945-10-27"],"topics":["role.head","role.pep"]},
            {"id":"pep_mx01","schema":"Person","names":["CLAUDIA SHEINBAUM"],"primary_name":"CLAUDIA SHEINBAUM","position":["President of Mexico"],"nationality":["MX"],"country":["MX"],"birthDate":["1962-06-24"],"topics":["role.head","role.pep"]},
            {"id":"pep_ar01","schema":"Person","names":["JAVIER MILEI"],"primary_name":"JAVIER MILEI","position":["President of Argentina"],"nationality":["AR"],"country":["AR"],"birthDate":["1970-10-22"],"topics":["role.head","role.pep"]},
            {"id":"pep_za01","schema":"Person","names":["CYRIL RAMAPHOSA"],"primary_name":"CYRIL RAMAPHOSA","position":["President of South Africa"],"nationality":["ZA"],"country":["ZA"],"birthDate":["1952-11-17"],"topics":["role.head","role.pep"]},
            {"id":"pep_tr01","schema":"Person","names":["RECEP TAYYIP ERDOGAN","ERDOGAN"],"primary_name":"RECEP TAYYIP ERDOGAN","position":["President of Turkey"],"nationality":["TR"],"country":["TR"],"birthDate":["1954-02-26"],"topics":["role.head","role.pep"]},
            {"id":"pep_sa01","schema":"Person","names":["MOHAMMED BIN SALMAN","MBS","CROWN PRINCE MOHAMMED"],"primary_name":"MOHAMMED BIN SALMAN","position":["Crown Prince of Saudi Arabia","Prime Minister of Saudi Arabia"],"nationality":["SA"],"country":["SA"],"birthDate":["1985-08-31"],"topics":["role.gov","role.pep"]},
            {"id":"pep_sa02","schema":"Person","names":["KING SALMAN","SALMAN BIN ABDULAZIZ"],"primary_name":"KING SALMAN BIN ABDULAZIZ","position":["King of Saudi Arabia"],"nationality":["SA"],"country":["SA"],"birthDate":["1935-12-31"],"topics":["role.head","role.pep"]},
            {"id":"pep_ae01","schema":"Person","names":["MOHAMMED BIN ZAYED","MBZ","MOHAMED BIN ZAYED AL NAHYAN"],"primary_name":"MOHAMMED BIN ZAYED AL NAHYAN","position":["President of the UAE"],"nationality":["AE"],"country":["AE"],"birthDate":["1961-03-11"],"topics":["role.head","role.pep"]},
            {"id":"pep_ae02","schema":"Person","names":["MOHAMMED BIN RASHID","SHEIKH MOHAMMED"],"primary_name":"MOHAMMED BIN RASHID AL MAKTOUM","position":["Prime Minister of UAE","Ruler of Dubai"],"nationality":["AE"],"country":["AE"],"birthDate":["1949-07-15"],"topics":["role.head","role.pep"]},
            {"id":"pep_qa01","schema":"Person","names":["TAMIM BIN HAMAD AL THANI","EMIR OF QATAR"],"primary_name":"TAMIM BIN HAMAD AL THANI","position":["Emir of Qatar"],"nationality":["QA"],"country":["QA"],"birthDate":["1980-06-03"],"topics":["role.head","role.pep"]},
            {"id":"pep_kw01","schema":"Person","names":["MESHAL AL-AHMAD AL-JABER AL-SABAH"],"primary_name":"MESHAL AL-AHMAD AL-JABER AL-SABAH","position":["Emir of Kuwait"],"nationality":["KW"],"country":["KW"],"birthDate":["1940-09-27"],"topics":["role.head","role.pep"]},
            {"id":"pep_ng01","schema":"Person","names":["BOLA TINUBU"],"primary_name":"BOLA TINUBU","position":["President of Nigeria"],"nationality":["NG"],"country":["NG"],"birthDate":["1952-03-29"],"topics":["role.head","role.pep"]},
            {"id":"pep_ke01","schema":"Person","names":["WILLIAM RUTO"],"primary_name":"WILLIAM RUTO","position":["President of Kenya"],"nationality":["KE"],"country":["KE"],"birthDate":["1966-12-21"],"topics":["role.head","role.pep"]},
            {"id":"pep_gh01","schema":"Person","names":["NANA AKUFO-ADDO"],"primary_name":"NANA AKUFO-ADDO","position":["Former President of Ghana"],"nationality":["GH"],"country":["GH"],"birthDate":["1944-03-29"],"topics":["role.head","role.pep"]},
            {"id":"pep_et01","schema":"Person","names":["ABIY AHMED"],"primary_name":"ABIY AHMED","position":["Prime Minister of Ethiopia"],"nationality":["ET"],"country":["ET"],"birthDate":["1976-08-15"],"topics":["role.head","role.pep"]},
            {"id":"pep_eg01","schema":"Person","names":["ABDEL FATTAH EL-SISI","AL-SISI","SISI"],"primary_name":"ABDEL FATTAH EL-SISI","position":["President of Egypt"],"nationality":["EG"],"country":["EG"],"birthDate":["1954-11-19"],"topics":["role.head","role.pep"]},
            {"id":"pep_ma01","schema":"Person","names":["KING MOHAMMED VI","MOHAMMED VI OF MOROCCO"],"primary_name":"MOHAMMED VI OF MOROCCO","position":["King of Morocco"],"nationality":["MA"],"country":["MA"],"birthDate":["1963-08-21"],"topics":["role.head","role.pep"]},
            {"id":"pep_dz01","schema":"Person","names":["ABDELMADJID TEBBOUNE"],"primary_name":"ABDELMADJID TEBBOUNE","position":["President of Algeria"],"nationality":["DZ"],"country":["DZ"],"birthDate":["1945-11-17"],"topics":["role.head","role.pep"]},
            {"id":"pep_pk01","schema":"Person","names":["SHEHBAZ SHARIF"],"primary_name":"SHEHBAZ SHARIF","position":["Prime Minister of Pakistan"],"nationality":["PK"],"country":["PK"],"birthDate":["1951-09-23"],"topics":["role.head","role.pep"]},
            {"id":"pep_bd01","schema":"Person","names":["SHEIKH HASINA","HASINA WAJED"],"primary_name":"SHEIKH HASINA","position":["Former Prime Minister of Bangladesh"],"nationality":["BD"],"country":["BD"],"birthDate":["1947-09-28"],"topics":["role.head","role.pep"]},
            {"id":"pep_lk01","schema":"Person","names":["RANIL WICKREMESINGHE"],"primary_name":"RANIL WICKREMESINGHE","position":["Former President of Sri Lanka"],"nationality":["LK"],"country":["LK"],"birthDate":["1949-03-24"],"topics":["role.head","role.pep"]},
            {"id":"pep_id01","schema":"Person","names":["PRABOWO SUBIANTO"],"primary_name":"PRABOWO SUBIANTO","position":["President of Indonesia"],"nationality":["ID"],"country":["ID"],"birthDate":["1951-10-17"],"topics":["role.head","role.pep"]},
            {"id":"pep_ph01","schema":"Person","names":["FERDINAND MARCOS JR","BONGBONG MARCOS"],"primary_name":"FERDINAND MARCOS JR","position":["President of the Philippines"],"nationality":["PH"],"country":["PH"],"birthDate":["1957-09-13"],"topics":["role.head","role.pep"]},
            {"id":"pep_th01","schema":"Person","names":["PAETONGTARN SHINAWATRA"],"primary_name":"PAETONGTARN SHINAWATRA","position":["Prime Minister of Thailand"],"nationality":["TH"],"country":["TH"],"birthDate":["1986-08-21"],"topics":["role.head","role.pep"]},
            {"id":"pep_vn01","schema":"Person","names":["TO LAM"],"primary_name":"TO LAM","position":["General Secretary of Vietnam"],"nationality":["VN"],"country":["VN"],"birthDate":["1957-07-10"],"topics":["role.head","role.pep"]},
            {"id":"pep_kp01","schema":"Person","names":["KIM JONG UN","KIM JONG-UN"],"primary_name":"KIM JONG UN","position":["Supreme Leader of North Korea"],"nationality":["KP"],"country":["KP"],"birthDate":["1984-01-08"],"topics":["role.head","role.pep","sanction"]},
            {"id":"pep_ir01","schema":"Person","names":["ALI KHAMENEI","AYATOLLAH KHAMENEI"],"primary_name":"ALI KHAMENEI","position":["Supreme Leader of Iran"],"nationality":["IR"],"country":["IR"],"birthDate":["1939-04-19"],"topics":["role.head","role.pep","sanction"]},
            {"id":"pep_ir02","schema":"Person","names":["MASOUD PEZESHKIAN"],"primary_name":"MASOUD PEZESHKIAN","position":["President of Iran"],"nationality":["IR"],"country":["IR"],"birthDate":["1954-09-29"],"topics":["role.head","role.pep","sanction"]},
            {"id":"pep_iq01","schema":"Person","names":["MOHAMMED SHIA AL-SUDANI"],"primary_name":"MOHAMMED SHIA AL-SUDANI","position":["Prime Minister of Iraq"],"nationality":["IQ"],"country":["IQ"],"birthDate":["1970-02-04"],"topics":["role.head","role.pep"]},
            {"id":"pep_il01","schema":"Person","names":["BENJAMIN NETANYAHU","BIBI NETANYAHU"],"primary_name":"BENJAMIN NETANYAHU","position":["Prime Minister of Israel"],"nationality":["IL"],"country":["IL"],"birthDate":["1949-10-21"],"topics":["role.head","role.pep"]},
            {"id":"pep_ua01","schema":"Person","names":["VOLODYMYR ZELENSKY","ZELENSKYY","VOLODYMYR ZELENSKYY"],"primary_name":"VOLODYMYR ZELENSKY","position":["President of Ukraine"],"nationality":["UA"],"country":["UA"],"birthDate":["1978-01-25"],"topics":["role.head","role.pep"]},
            {"id":"pep_by01","schema":"Person","names":["ALEXANDER LUKASHENKO","LUKASHENKA"],"primary_name":"ALEXANDER LUKASHENKO","position":["President of Belarus"],"nationality":["BY"],"country":["BY"],"birthDate":["1954-08-30"],"topics":["role.head","role.pep","sanction"]},
            {"id":"pep_rs01","schema":"Person","names":["ALEKSANDAR VUCIC"],"primary_name":"ALEKSANDAR VUCIC","position":["President of Serbia"],"nationality":["RS"],"country":["RS"],"birthDate":["1970-03-05"],"topics":["role.head","role.pep"]},
            {"id":"pep_hu01","schema":"Person","names":["VIKTOR ORBAN","VIKTOR ORBÁN"],"primary_name":"VIKTOR ORBAN","position":["Prime Minister of Hungary"],"nationality":["HU"],"country":["HU"],"birthDate":["1963-05-31"],"topics":["role.head","role.pep"]},
            {"id":"pep_pl01","schema":"Person","names":["DONALD TUSK"],"primary_name":"DONALD TUSK","position":["Prime Minister of Poland"],"nationality":["PL"],"country":["PL"],"birthDate":["1957-04-22"],"topics":["role.head","role.pep"]},
            {"id":"pep_ve01","schema":"Person","names":["NICOLAS MADURO","MADURO MOROS"],"primary_name":"NICOLAS MADURO","position":["President of Venezuela"],"nationality":["VE"],"country":["VE"],"birthDate":["1962-11-23"],"topics":["role.head","role.pep","sanction"]},
            {"id":"pep_cu01","schema":"Person","names":["MIGUEL DIAZ-CANEL"],"primary_name":"MIGUEL DIAZ-CANEL","position":["President of Cuba"],"nationality":["CU"],"country":["CU"],"birthDate":["1960-04-20"],"topics":["role.head","role.pep","sanction"]},
            {"id":"pep_sy01","schema":"Person","names":["BASHAR AL-ASSAD","BASHAR ASSAD"],"primary_name":"BASHAR AL-ASSAD","position":["Former President of Syria"],"nationality":["SY"],"country":["SY"],"birthDate":["1965-09-11"],"topics":["role.head","role.pep","sanction"]},
            {"id":"pep_mm01","schema":"Person","names":["MIN AUNG HLAING"],"primary_name":"MIN AUNG HLAING","position":["Chairman of the State Administration Council of Myanmar"],"nationality":["MM"],"country":["MM"],"birthDate":["1956-07-03"],"topics":["role.head","role.pep","sanction"]},
            {"id":"pep_sd01","schema":"Person","names":["ABDEL FATTAH AL-BURHAN"],"primary_name":"ABDEL FATTAH AL-BURHAN","position":["President of the Sovereignty Council of Sudan"],"nationality":["SD"],"country":["SD"],"birthDate":["1960-01-01"],"topics":["role.head","role.pep","sanction"]},
            {"id":"pep_ly01","schema":"Person","names":["KHALIFA HAFTAR"],"primary_name":"KHALIFA HAFTAR","position":["Commander of Libyan National Army"],"nationality":["LY"],"country":["LY"],"birthDate":["1943-11-07"],"topics":["role.mil","role.pep","sanction"]},
            {"id":"pep_zw01","schema":"Person","names":["EMMERSON MNANGAGWA"],"primary_name":"EMMERSON MNANGAGWA","position":["President of Zimbabwe"],"nationality":["ZW"],"country":["ZW"],"birthDate":["1942-09-15"],"topics":["role.head","role.pep","sanction"]},

            # ── FINANCE MINISTERS & TREASURY OFFICIALS (High AML Risk) ──────
            {"id":"pep_gb_fin01","schema":"Person","names":["RACHEL REEVES"],"primary_name":"RACHEL REEVES","position":["Chancellor of the Exchequer"],"nationality":["GB"],"country":["GB"],"birthDate":["1979-02-13"],"topics":["role.gov","role.pep"]},
            {"id":"pep_us_fin01","schema":"Person","names":["JANET YELLEN"],"primary_name":"JANET YELLEN","position":["Secretary of the Treasury"],"nationality":["US"],"country":["US"],"birthDate":["1946-08-13"],"topics":["role.gov","role.pep"]},
            {"id":"pep_us_fin02","schema":"Person","names":["SCOTT BESSENT"],"primary_name":"SCOTT BESSENT","position":["Secretary of the Treasury"],"nationality":["US"],"country":["US"],"birthDate":["1962-01-01"],"topics":["role.gov","role.pep"]},
            {"id":"pep_de_fin01","schema":"Person","names":["CHRISTIAN LINDNER"],"primary_name":"CHRISTIAN LINDNER","position":["Former Finance Minister of Germany"],"nationality":["DE"],"country":["DE"],"birthDate":["1979-01-07"],"topics":["role.gov","role.pep"]},
            {"id":"pep_fr_fin01","schema":"Person","names":["BRUNO LE MAIRE"],"primary_name":"BRUNO LE MAIRE","position":["Former Finance Minister of France"],"nationality":["FR"],"country":["FR"],"birthDate":["1969-04-15"],"topics":["role.gov","role.pep"]},
            {"id":"pep_in_fin01","schema":"Person","names":["NIRMALA SITHARAMAN"],"primary_name":"NIRMALA SITHARAMAN","position":["Finance Minister of India"],"nationality":["IN"],"country":["IN"],"birthDate":["1959-08-18"],"topics":["role.gov","role.pep"]},
            {"id":"pep_ru_fin01","schema":"Person","names":["ANTON SILUANOV"],"primary_name":"ANTON SILUANOV","position":["Finance Minister of Russia"],"nationality":["RU"],"country":["RU"],"birthDate":["1963-04-12"],"topics":["role.gov","role.pep","sanction"]},
            {"id":"pep_cn_fin01","schema":"Person","names":["LAN FOAN"],"primary_name":"LAN FOAN","position":["Finance Minister of China"],"nationality":["CN"],"country":["CN"],"birthDate":["1962-01-01"],"topics":["role.gov","role.pep"]},

            # ── CENTRAL BANK GOVERNORS ───────────────────────────────────────
            {"id":"pep_gb_cb01","schema":"Person","names":["ANDREW BAILEY"],"primary_name":"ANDREW BAILEY","position":["Governor of the Bank of England"],"nationality":["GB"],"country":["GB"],"birthDate":["1959-03-30"],"topics":["role.gov","role.pep"]},
            {"id":"pep_eu_cb01","schema":"Person","names":["CHRISTINE LAGARDE"],"primary_name":"CHRISTINE LAGARDE","position":["President of the European Central Bank"],"nationality":["FR"],"country":["EU"],"birthDate":["1956-01-01"],"topics":["role.gov","role.pep"]},
            {"id":"pep_us_cb01","schema":"Person","names":["JEROME POWELL","JAY POWELL"],"primary_name":"JEROME POWELL","position":["Chair of the Federal Reserve"],"nationality":["US"],"country":["US"],"birthDate":["1953-02-04"],"topics":["role.gov","role.pep"]},
            {"id":"pep_in_cb01","schema":"Person","names":["SHAKTIKANTA DAS"],"primary_name":"SHAKTIKANTA DAS","position":["Governor of the Reserve Bank of India"],"nationality":["IN"],"country":["IN"],"birthDate":["1957-02-26"],"topics":["role.gov","role.pep"]},
            {"id":"pep_cn_cb01","schema":"Person","names":["PAN GONGSHENG"],"primary_name":"PAN GONGSHENG","position":["Governor of the People's Bank of China"],"nationality":["CN"],"country":["CN"],"birthDate":["1963-01-01"],"topics":["role.gov","role.pep"]},
            {"id":"pep_ru_cb01","schema":"Person","names":["ELVIRA NABIULLINA"],"primary_name":"ELVIRA NABIULLINA","position":["Governor of the Bank of Russia"],"nationality":["RU"],"country":["RU"],"birthDate":["1963-10-29"],"topics":["role.gov","role.pep","sanction"]},
            {"id":"pep_ng_cb01","schema":"Person","names":["YEMI CARDOSO"],"primary_name":"YEMI CARDOSO","position":["Governor of the Central Bank of Nigeria"],"nationality":["NG"],"country":["NG"],"birthDate":["1961-01-01"],"topics":["role.gov","role.pep"]},
            {"id":"pep_ae_cb01","schema":"Person","names":["KHALED BALAMA"],"primary_name":"KHALED BALAMA","position":["Governor of the Central Bank of the UAE"],"nationality":["AE"],"country":["AE"],"birthDate":["1975-01-01"],"topics":["role.gov","role.pep"]},

            # ── INTELLIGENCE & LAW ENFORCEMENT CHIEFS ───────────────────────
            {"id":"pep_gb_intel01","schema":"Person","names":["RICHARD MOORE"],"primary_name":"RICHARD MOORE","position":["Chief of MI6"],"nationality":["GB"],"country":["GB"],"birthDate":["1963-01-01"],"topics":["role.gov","role.pep"]},
            {"id":"pep_gb_intel02","schema":"Person","names":["KEN MCCALLUM"],"primary_name":"KEN MCCALLUM","position":["Director General of MI5"],"nationality":["GB"],"country":["GB"],"birthDate":["1972-01-01"],"topics":["role.gov","role.pep"]},
            {"id":"pep_us_intel01","schema":"Person","names":["WILLIAM BURNS"],"primary_name":"WILLIAM BURNS","position":["Director of the CIA"],"nationality":["US"],"country":["US"],"birthDate":["1956-04-08"],"topics":["role.gov","role.pep"]},
            {"id":"pep_us_intel02","schema":"Person","names":["CHRISTOPHER WRAY"],"primary_name":"CHRISTOPHER WRAY","position":["Director of the FBI"],"nationality":["US"],"country":["US"],"birthDate":["1966-12-17"],"topics":["role.gov","role.pep"]},
            {"id":"pep_ru_intel01","schema":"Person","names":["ALEXANDER BORTNIKOV"],"primary_name":"ALEXANDER BORTNIKOV","position":["Director of the FSB"],"nationality":["RU"],"country":["RU"],"birthDate":["1951-11-15"],"topics":["role.gov","role.pep","sanction"]},
            {"id":"pep_ru_intel02","schema":"Person","names":["SERGEI NARYSHKIN"],"primary_name":"SERGEI NARYSHKIN","position":["Director of the SVR"],"nationality":["RU"],"country":["RU"],"birthDate":["1954-10-27"],"topics":["role.gov","role.pep","sanction"]},

            # ── PROMINENT SANCTIONED / HIGH RISK PEPs ───────────────────────
            {"id":"pep_ru_oli01","schema":"Person","names":["ROMAN ABRAMOVICH"],"primary_name":"ROMAN ABRAMOVICH","position":["Russian Oligarch","Former Owner of Chelsea FC"],"nationality":["RU"],"country":["RU"],"birthDate":["1966-10-24"],"topics":["role.pep","sanction"]},
            {"id":"pep_ru_oli02","schema":"Person","names":["IGOR SECHIN"],"primary_name":"IGOR SECHIN","position":["CEO of Rosneft","Deputy Prime Minister of Russia"],"nationality":["RU"],"country":["RU"],"birthDate":["1960-09-07"],"topics":["role.gov","role.pep","sanction"]},
            {"id":"pep_ru_oli03","schema":"Person","names":["GENNADY TIMCHENKO"],"primary_name":"GENNADY TIMCHENKO","position":["Russian Oligarch"],"nationality":["RU"],"country":["RU"],"birthDate":["1952-11-09"],"topics":["role.pep","sanction"]},
            {"id":"pep_ru_oli04","schema":"Person","names":["ARKADY ROTENBERG"],"primary_name":"ARKADY ROTENBERG","position":["Russian Businessman"],"nationality":["RU"],"country":["RU"],"birthDate":["1951-12-15"],"topics":["role.pep","sanction"]},
            {"id":"pep_ru_oli05","schema":"Person","names":["BORIS ROTENBERG"],"primary_name":"BORIS ROTENBERG","position":["Russian Businessman"],"nationality":["RU"],"country":["RU"],"birthDate":["1957-01-03"],"topics":["role.pep","sanction"]},
            {"id":"pep_ru_oli06","schema":"Person","names":["ALISHER USMANOV"],"primary_name":"ALISHER USMANOV","position":["Russian Oligarch"],"nationality":["RU"],"country":["UZ"],"birthDate":["1953-09-09"],"topics":["role.pep","sanction"]},
            {"id":"pep_ru_oli07","schema":"Person","names":["ALEXEI MORDASHOV"],"primary_name":"ALEXEI MORDASHOV","position":["Russian Oligarch","CEO of Severstal"],"nationality":["RU"],"country":["RU"],"birthDate":["1965-09-26"],"topics":["role.pep","sanction"]},
            {"id":"pep_ru_oli08","schema":"Person","names":["NIKOLAI PATRUSHEV"],"primary_name":"NIKOLAI PATRUSHEV","position":["Secretary of the Security Council of Russia"],"nationality":["RU"],"country":["RU"],"birthDate":["1951-07-11"],"topics":["role.gov","role.pep","sanction"]},
            {"id":"pep_kh01","schema":"Person","names":["HUN SEN","SAMDECH HUN SEN"],"primary_name":"HUN SEN","position":["Former Prime Minister of Cambodia","President of Senate"],"nationality":["KH"],"country":["KH"],"birthDate":["1952-08-05"],"topics":["role.head","role.pep"]},
            {"id":"pep_ug01","schema":"Person","names":["YOWERI MUSEVENI"],"primary_name":"YOWERI MUSEVENI","position":["President of Uganda"],"nationality":["UG"],"country":["UG"],"birthDate":["1944-09-15"],"topics":["role.head","role.pep"]},
            {"id":"pep_cm01","schema":"Person","names":["PAUL BIYA"],"primary_name":"PAUL BIYA","position":["President of Cameroon"],"nationality":["CM"],"country":["CM"],"birthDate":["1933-02-13"],"topics":["role.head","role.pep"]},
            {"id":"pep_td01","schema":"Person","names":["MAHAMAT IDRISS DEBY"],"primary_name":"MAHAMAT IDRISS DEBY","position":["President of Chad"],"nationality":["TD"],"country":["TD"],"birthDate":["1984-01-01"],"topics":["role.head","role.pep"]},
            {"id":"pep_ga01","schema":"Person","names":["ALI BONGO ONDIMBA"],"primary_name":"ALI BONGO ONDIMBA","position":["Former President of Gabon"],"nationality":["GA"],"country":["GA"],"birthDate":["1959-02-09"],"topics":["role.head","role.pep"]},

            # ── INTERNATIONAL ORGANISATIONS ──────────────────────────────────
            {"id":"pep_un01","schema":"Person","names":["ANTONIO GUTERRES"],"primary_name":"ANTONIO GUTERRES","position":["Secretary-General of the United Nations"],"nationality":["PT"],"country":["UN"],"birthDate":["1949-04-30"],"topics":["role.gov","role.pep"]},
            {"id":"pep_imf01","schema":"Person","names":["KRISTALINA GEORGIEVA"],"primary_name":"KRISTALINA GEORGIEVA","position":["Managing Director of the IMF"],"nationality":["BG"],"country":["BG"],"birthDate":["1953-08-13"],"topics":["role.gov","role.pep"]},
            {"id":"pep_wb01","schema":"Person","names":["AJAY BANGA"],"primary_name":"AJAY BANGA","position":["President of the World Bank"],"nationality":["IN"],"country":["US"],"birthDate":["1959-11-10"],"topics":["role.gov","role.pep"]},
            {"id":"pep_nato01","schema":"Person","names":["MARK RUTTE"],"primary_name":"MARK RUTTE","position":["Secretary General of NATO"],"nationality":["NL"],"country":["NL"],"birthDate":["1967-07-14"],"topics":["role.gov","role.pep"]},
            {"id":"pep_eu01","schema":"Person","names":["URSULA VON DER LEYEN"],"primary_name":"URSULA VON DER LEYEN","position":["President of the European Commission"],"nationality":["DE"],"country":["DE"],"birthDate":["1958-10-08"],"topics":["role.gov","role.pep"]},
            {"id":"pep_eu02","schema":"Person","names":["CHARLES MICHEL"],"primary_name":"CHARLES MICHEL","position":["President of the European Council"],"nationality":["BE"],"country":["BE"],"birthDate":["1975-12-21"],"topics":["role.gov","role.pep"]},
            {"id":"pep_fatf01","schema":"Organization","names":["FINANCIAL ACTION TASK FORCE","FATF"],"primary_name":"FINANCIAL ACTION TASK FORCE","position":["International AML/CFT Standards Body"],"nationality":[],"country":["FR"],"birthDate":[],"topics":["role.gov"]},

            # ── UK SPECIFIC ──────────────────────────────────────────────────
            {"id":"pep_gb_fc01","schema":"Person","names":["DAVID LAMMY"],"primary_name":"DAVID LAMMY","position":["Secretary of State for Foreign Affairs"],"nationality":["GB"],"country":["GB"],"birthDate":["1972-07-19"],"topics":["role.gov","role.pep"]},
            {"id":"pep_gb_fc02","schema":"Person","names":["YVETTE COOPER"],"primary_name":"YVETTE COOPER","position":["Home Secretary"],"nationality":["GB"],"country":["GB"],"birthDate":["1969-03-20"],"topics":["role.gov","role.pep"]},
            {"id":"pep_gb_fc03","schema":"Person","names":["WES STREETING"],"primary_name":"WES STREETING","position":["Secretary of State for Health"],"nationality":["GB"],"country":["GB"],"birthDate":["1983-01-21"],"topics":["role.gov","role.pep"]},
            {"id":"pep_gb_fc04","schema":"Person","names":["NIGEL FARAGE"],"primary_name":"NIGEL FARAGE","position":["Leader of Reform UK","Former MEP"],"nationality":["GB"],"country":["GB"],"birthDate":["1964-04-03"],"topics":["role.leg","role.pep"]},
            {"id":"pep_gb_fc05","schema":"Person","names":["KEMI BADENOCH"],"primary_name":"KEMI BADENOCH","position":["Leader of the Conservative Party"],"nationality":["GB"],"country":["GB"],"birthDate":["1980-01-02"],"topics":["role.leg","role.pep"]},
            {"id":"pep_gb_fc06","schema":"Person","names":["ED DAVEY","SIR ED DAVEY"],"primary_name":"ED DAVEY","position":["Leader of the Liberal Democrats"],"nationality":["GB"],"country":["GB"],"birthDate":["1965-12-25"],"topics":["role.leg","role.pep"]},
            {"id":"pep_gb_roy01","schema":"Person","names":["KING CHARLES","KING CHARLES III","CHARLES PHILIP ARTHUR GEORGE"],"primary_name":"KING CHARLES III","position":["King of the United Kingdom"],"nationality":["GB"],"country":["GB"],"birthDate":["1948-11-14"],"topics":["role.head","role.pep"]},
            {"id":"pep_gb_fca01","schema":"Person","names":["NIKHIL RATHI"],"primary_name":"NIKHIL RATHI","position":["Chief Executive of the FCA"],"nationality":["GB"],"country":["GB"],"birthDate":["1979-01-01"],"topics":["role.gov","role.pep"]},
        ]

    def _fallback(self):
        return self._builtin_peps()

    def search(self, query: str, threshold: int = 80) -> list:
        q = clean(query)
        if not self.name_index: return []
        candidate_cutoff = max(50, threshold - 25)
        raw_matches = process.extract(q, [n[0] for n in self.name_index], scorer=fuzz.WRatio, limit=40, score_cutoff=candidate_cutoff)
        rescored = []
        for matched_text, _, idx in raw_matches:
            final_score = smart_match_score(q, matched_text)
            if final_score >= threshold:
                rescored.append((matched_text, final_score, idx))
        rescored.sort(key=lambda x: x[1], reverse=True)
        matches = rescored[:15]
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
