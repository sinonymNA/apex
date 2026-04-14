# Apex Trading System

Automated paper trading system for SPY. Three Railway services: FastAPI backend, Python worker (APScheduler), and React dashboard.

```
PostgreSQL ──┬── worker/trader.py  (APScheduler, Mon–Fri 9:25–4:05 ET)
             └── api/main.py       (FastAPI, 7 REST routes)
                        ▲
               dashboard/ (React, mobile-first, auto-refresh 30s)
```

---

## Prerequisites

- Python 3.11+
- Node 18+ (for dashboard)
- PostgreSQL (or SQLite for local dev — automatic fallback)

---

## Local Setup

```bash
# 1. Clone and enter repo
git clone <repo-url> && cd apex

# 2. Copy env file and fill in values
cp .env.example .env
# edit .env — set ALPACA_API_KEY, ALPACA_SECRET_KEY, GMAIL_*, DASHBOARD_SECRET

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run backtest (trains regime model, saves logs/backtest_report.json)
python backtest/run.py

# 5. Start API server
uvicorn api.main:app --reload

# 6. Start worker (separate terminal)
python worker/trader.py

# 7. Start dashboard (separate terminal)
cd dashboard && npm install && npm start
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ALPACA_API_KEY` | Yes | Alpaca paper trading API key |
| `ALPACA_SECRET_KEY` | Yes | Alpaca paper trading secret |
| `ALPACA_BASE_URL` | No | Defaults to paper trading URL |
| `ANTHROPIC_API_KEY` | No | Enables AI anomaly diagnosis (Phase 1) |
| `GMAIL_USER` | Yes | Gmail address for daily reports |
| `GMAIL_APP_PASSWORD` | Yes | Gmail app password (not login password) |
| `NOTIFY_EMAIL` | Yes | Recipient email for daily reports |
| `DATABASE_URL` | No | PostgreSQL URL; falls back to SQLite |
| `DASHBOARD_SECRET` | Yes | Shared secret for dashboard auth |
| `REACT_APP_API_URL` | No | API base URL for dashboard build |

---

## Railway Deployment

1. Create a **PostgreSQL** plugin in Railway — note the `DATABASE_URL`.
2. Create **three Railway services**, all pointing at this GitHub repo.
3. Configure each service:

   **Service 1 — api** (uses `railway.toml` defaults, no override needed)

   **Service 2 — worker**
   ```
   RAILWAY_RUN_COMMAND = python worker/trader.py
   ```

   **Service 3 — dashboard**
   ```
   RAILWAY_RUN_COMMAND = cd dashboard && npm install && npm run build && npx serve -s build -l $PORT
   REACT_APP_API_URL = https://your-api-service.railway.app
   ```

4. Add all env vars from the table above to **each** service.
5. Deploy. The worker and API share the same `DATABASE_URL`.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Daily Operator Checklist

1. Read 4:30 PM email — check P&L, gate status, anomalies.
2. Check dashboard URL — verify RUNNING badge and heartbeat.
3. Note gate day count — stop at Day 20, switch to ES after passing all 5 gates.
