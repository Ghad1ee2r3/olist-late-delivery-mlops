# Task 03: Olist late delivery inference service

Notebooks in Task[02] already train the model. This folder turns that into a service: config, modules, DVC, MLflow, tests, FastAPI, Docker, CI, monitoring.

**Use Python 3.9** (same version that trained the model pickle).

Reference: https://madewithml.com

```bash
cd "Task[03]"
```

---

## Quick start (run the app)

```bash
python3.9 -m pip install -r requirements.txt
python3.9 -m src.register_model
python3.9 -m app.run
```

Open: http://127.0.0.1:8000/docs

| check | command / what you should see |
|---|---|
| health | `curl -s http://127.0.0.1:8000/health` → `"status":"ok"` |
| model | `curl -s http://127.0.0.1:8000/model` → `gbdt_weighted`, version, `n_features`: 63 |
| predict | POST `/predict` with sample body (drop `is_late`) → `on_time`, ~0.3356 |
| metrics | `curl -s http://127.0.0.1:8000/metrics` → request counts and late mix |

Sample predict:

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @<(python3.9 -c "import json;d=json.load(open('data/sample_order.json'));d.pop('is_late',None);print(json.dumps(d))")
```

Batch:

```bash
curl -s -X POST "http://127.0.0.1:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d @<(python3.9 -c "import json;d=json.load(open('data/sample_order.json'));d.pop('is_late',None);print(json.dumps({'orders':[d,d]}))")
```

---

## Folder map

| folder / file | why |
|---|---|
| `config/` | settings, expectations, alert notes |
| `src/` | inference modules (no training) |
| `app/` | FastAPI routes |
| `data/` | sample order + notebook metrics for register |
| `models/` | transformers (+ joblib used only to register once) |
| `mlruns/` | local MLflow store (created by register) |
| `tests/` | pytest |
| `logs/` | `service.log` + `predictions.jsonl` |
| `Dockerfile` + `docker-compose.yml` | API + Postgres + artifact mounts |

---

## Step by step (how to run + how to check)

### 0) Install

```bash
python3.9 -m pip install -r requirements.txt
python3.9 -m pip install -r requirements-dev.txt
```

| check | expected |
|---|---|
| `python3.9 -c "import sklearn,mlflow,fastapi; print('ok')"` | prints `ok` |

---

### 1) Config

```bash
python3.9 -m src.check_config
```

| check | expected |
|---|---|
| printed project name | `olist_late_delivery` |
| label | `is_late` |
| change `project.name` in yaml, run again | printed name changes |

---

### 2) Modules match notebook 5

```bash
python3.9 -m src.run_predict
python3.9 -m src.check_match
```

| check | expected |
|---|---|
| `run_predict` | `on_time`, probability ~0.3356 |
| `check_match` | `match ok`, max abs diff ~0.0, shape (20, 63) |

---

### 3) Logging and bad input

```bash
python3.9 -m src.run_predict
python3.9 -m src.try_bad_input
```

| check | expected |
|---|---|
| `logs/service.log` | INFO for good order, ERROR for missing columns, WARNING for null weight |
| bad input | process stays up, returns `ok: false` |

---

### 4) DVC + input expectations

```bash
dvc status
python3.9 -m src.check_expectations
```

| check | expected |
|---|---|
| `dvc status` | data up to date (if DVC was set up) |
| expectations script | sample accepted, negative price rejected |

Studio account steps (optional): `DVC_ACCOUNT.md`

---

### 5) MLflow registry

```bash
python3.9 -m src.register_model
python3.9 -m src.register_model
python3.9 -m src.run_predict
```

| check | expected |
|---|---|
| first register | creates `late_delivery` version 1, stage Production |
| second register | says already registered (no new version) |
| predict result | includes `model_version` from registry (e.g. `"1"`) |
| folder | `mlruns/` exists |

---

### 6) Tests (pytest)

```bash
python3.9 -m src.register_model
python3.9 -m pytest
```

| check | expected |
|---|---|
| summary | about **28 passed** |
| break one assert on purpose | pytest fails (red) |

Lint / format (CI tools):

```bash
ruff check app src tests
ruff format --check app src tests
```

| check | expected |
|---|---|
| both commands | clean / already formatted |

---

### 7) FastAPI (local)

```bash
python3.9 -m src.register_model
python3.9 -m app.run
```

| route | how to check |
|---|---|
| `GET /health` | status ok |
| `GET /model` | name, version, stage, 63 features |
| `POST /predict` | one order → label + probability + version |
| `POST /predict/batch` | many orders |
| `GET /docs` | Swagger UI loads (needs pydantic 2.10.x) |
| bad JSON | 422 |
| negative `items_price_sum` | 400 |

---

### 8) Docker

Start Docker Desktop first. Keep a few GB free on disk.

```bash
cp .env.example .env
# set POSTGRES_PASSWORD in .env (same value inside DATABASE_URL)
docker compose up --build
```

| check | expected |
|---|---|
| `curl -s http://127.0.0.1:8000/health` | ok |
| `curl -s http://127.0.0.1:8000/model` | model info |
| docs | http://127.0.0.1:8000/docs |
| stop | `docker compose down` |

Need `mlruns/` (registered) and `models/` mounted into the container.

If build fails with read-only / I/O errors: free disk space, restart Docker, then `docker compose build --no-cache`.

---

### 9) CI/CD

Local (same checks as GitHub Actions):

```bash
ruff check app src tests
ruff format --check app src tests
python3.9 -m src.register_model
python3.9 -m pytest
```

Hooks (needs git):

```bash
pre-commit install
pre-commit run --all-files
```

| check | expected |
|---|---|
| hooks / Actions | lint + format + tests must pass |
| failing test | pipeline stops (red) |

Workflows:
- `Task[03]/.github/workflows/ci.yml` if this folder is the repo
- `../.github/workflows/task03-ci.yml` if project1 is the repo

---

### 10) Monitoring

With the API running:

```bash
# score one order (see Quick start curl)
curl -s http://127.0.0.1:8000/metrics
```

| check | where / expected |
|---|---|
| live metrics | `GET /metrics` → `requests_total`, `latency_ms_avg`, `error_rate`, `predictions.late_rate` |
| drift hint | `drift.baseline_late_rate` vs `current_late_rate` |
| alerts list | `alerts_triggered` (empty until thresholds hit) |
| prediction store | `logs/predictions.jsonl` gains one JSON line per scored order |
| alert decisions | read `config/ALERTS.md` |
| thresholds | `monitoring:` block in `config/config.yaml` |

`actual_is_late` in the jsonl file stays `null` until the real delivery date is known. Then you can score the model offline.

---

## Routes summary

| method | path | job |
|---|---|---|
| GET | `/health` | process up |
| GET | `/model` | registry name / version / stage |
| GET | `/metrics` | counts, latency, errors, prediction mix |
| POST | `/predict` | one order |
| POST | `/predict/batch` | many orders |

---

## Common problems

| problem | fix |
|---|---|
| wrong Python / missing joblib | use `python3.9` |
| `mlflow store missing` | `python3.9 -m src.register_model` |
| `/docs` OpenAPI 500 | pin `pydantic==2.10.6` |
| Docker disk full / I/O error | free space, restart Docker Desktop |
| `pip` fails on dvc/pygit2 | skip dvc for now (not in `requirements-dev.txt`); needs Xcode for pygit2 |
| predict returns 405 in browser | use POST, not GET |


