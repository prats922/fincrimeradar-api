# FinCrimeRadar API

Free, open-source sanctions, PEP and adverse media screening API.

## Endpoints

### Screen a name
```
GET /api/screen?q=NAME&type=all&threshold=80
```

**Parameters:**
- `q` — name or entity to screen (required)
- `type` — `all` | `sanctions` | `pep` | `adverse` (default: all)
- `threshold` — match confidence 50–100 (default: 80)

**Example:**
```
GET /api/screen?q=Vladimir+Putin&type=all&threshold=85
```

### Health check
```
GET /api/health
```

## Data Sources

- **Sanctions:** OpenSanctions (OFAC, UN, EU, OFSI, and 40+ lists)
- **PEP:** OpenSanctions PEP dataset
- **Adverse Media:** GDELT Project (global news)

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

API available at `http://localhost:8000`

## Disclaimer

For educational purposes only. Not a substitute for regulated compliance screening.
