# Apex Trading System

Automated SPY paper trading system. One Railway service runs everything: dashboard, API, and trading worker.

---

## Deploy to Railway (5 steps, no code)

### Step 1 — Create a Railway project
Go to **[railway.app](https://railway.app)** → **New Project** → **Deploy from GitHub repo** → select `sinonymNA/apex`

### Step 2 — Add environment variables
In Railway → your service → **Variables** tab, add these:

| Variable | What to put |
|---|---|
| `DASHBOARD_SECRET` | Any password you want (you'll use this to log into the dashboard) |
| `ALPACA_API_KEY` | From alpaca.markets → Paper Trading → API Keys |
| `ALPACA_SECRET_KEY` | Same place as above |
| `GMAIL_USER` | Your Gmail address |
| `GMAIL_APP_PASSWORD` | [Create one here](https://myaccount.google.com/apppasswords) (not your Gmail password) |
| `NOTIFY_EMAIL` | Where the 4:30 PM daily report goes |

### Step 3 — Deploy
Click **Deploy**. Railway builds the dashboard, installs dependencies, and starts everything.

### Step 4 — Open your dashboard
Find your Railway URL (shown in the service panel) and open it. Log in with your `DASHBOARD_SECRET`.

### Step 5 — Add PostgreSQL (optional but recommended)
In Railway → **New** → **Database** → **PostgreSQL**. Railway automatically sets `DATABASE_URL` — no configuration needed. Without it, the system uses SQLite (data resets on redeploy).

---

## What happens automatically on first deploy

1. Python dependencies installed
2. React dashboard built
3. Database tables created
4. Regime classifier trained (uses synthetic data if market is closed)
5. Trading worker starts — waits for next trading session

---

## Daily operator checklist (3 steps)

1. Read 4:30 PM email — check P&L, gate status, any anomalies
2. Check dashboard — verify green RUNNING badge and recent heartbeat
3. Note gate day count — 20 trading days is the evaluation period

---

## Local development

```bash
cp .env.example .env   # fill in your keys
pip install -r requirements.txt
uvicorn api.main:app --reload   # starts API + worker + serves dashboard at localhost:8000
```

Run tests:
```bash
pytest tests/ -v
```
