"""
worker/db.py — Database layer for Sable Stocks.

SQLAlchemy 2.0. Reads DATABASE_URL from env.
Falls back to sqlite:///./logs/trades.db if not set.
All tables are created on import (CREATE TABLE IF NOT EXISTS).
"""
import os
import json
from datetime import datetime, date, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean,
    DateTime, Date, Text, JSON, inspect as sa_inspect, text
)
from sqlalchemy.orm import DeclarativeBase, Session
from loguru import logger

load_dotenv()

# ── Engine setup ──────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if not DATABASE_URL:
    _log_dir = Path(__file__).parent.parent / "logs"
    _log_dir.mkdir(exist_ok=True)
    DATABASE_URL = f"sqlite:///{_log_dir}/trades.db"

if "postgresql" in DATABASE_URL:
    # Railway internal network doesn't need SSL; public proxy may.
    # "prefer" tries SSL and falls back gracefully — works for both.
    _ssl_mode = "prefer" if "sslmode" not in DATABASE_URL else None
    _connect_args = {"sslmode": _ssl_mode} if _ssl_mode else {}
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args=_connect_args,
    )
    logger.info("PostgreSQL engine created with pool_pre_ping")
else:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
    logger.warning(
        "Running on SQLite — data will be lost on Railway redeploy. "
        "Add Railway PostgreSQL plugin for persistence."
    )


# ── ORM Base ──────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Models ────────────────────────────────────────────────────────────────────
class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_time = Column(DateTime, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    symbol = Column(String(16), nullable=False, default="SPY")
    direction = Column(String(8), nullable=False, default="LONG")
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    shares = Column(Integer, nullable=True)
    pnl_dollars = Column(Float, nullable=True)
    pnl_r = Column(Float, nullable=True)
    atr_at_entry = Column(Float, nullable=True)
    volume_ratio = Column(Float, nullable=True)
    exit_reason = Column(String(64), nullable=True)
    regime = Column(String(32), nullable=True)
    consecutive_losses = Column(Integer, nullable=True)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class SystemStatus(Base):
    __tablename__ = "system_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    status = Column(String(16), default="RUNNING")
    regime = Column(String(32), nullable=True)
    trade_count_today = Column(Integer, default=0)
    daily_pnl = Column(Float, default=0.0)
    consecutive_losses = Column(Integer, default=0)
    kill_switch_active = Column(Boolean, default=False)
    session_day = Column(Integer, default=1)
    message = Column(Text, nullable=True)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class DailySummary(Base):
    __tablename__ = "daily_summary"

    trade_date = Column(Date, primary_key=True, default=date.today)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    gross_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    avg_r = Column(Float, default=0.0)
    rule_violations = Column(Integer, default=0)
    regime_distribution = Column(Text, nullable=True)  # JSON string

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        if d.get("regime_distribution"):
            try:
                d["regime_distribution"] = json.loads(d["regime_distribution"])
            except Exception:
                pass
        return d


class RiskCheck(Base):
    __tablename__ = "risk_checks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    checked_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    check_type = Column(String(64), nullable=False)
    result = Column(String(16), nullable=False)  # APPROVED / BLOCKED
    reason = Column(Text, nullable=True)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class GateStatus(Base):
    __tablename__ = "gate_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    checked_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    day_number = Column(Integer, default=0)
    gate1_return = Column(Float, nullable=True)
    gate2_violations = Column(Integer, nullable=True)
    gate3_drawdown = Column(Float, nullable=True)
    gate4_winrate = Column(Float, nullable=True)
    gate5_slippage = Column(Float, nullable=True)
    all_passed = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    detected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    event_type = Column(String(64), nullable=True)
    severity = Column(String(16), default="INFO")  # CRITICAL / WARNING / INFO
    diagnosis = Column(Text, nullable=True)
    recommendation = Column(Text, nullable=True)
    context_json = Column(Text, nullable=True)  # JSON string

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        if d.get("context_json"):
            try:
                d["context_json"] = json.loads(d["context_json"])
            except Exception:
                pass
        return d


class NearMissSignal(Base):
    __tablename__ = "near_miss_signals"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    timestamp           = Column(DateTime, nullable=False,
                                 default=lambda: datetime.now(timezone.utc))
    symbol              = Column(String(16), nullable=False, default="SPY")
    close               = Column(Float, nullable=True)
    breakout_level      = Column(Float, nullable=True)
    percent_to_breakout = Column(Float, nullable=True)  # negative = below level
    volume              = Column(Float, nullable=True)
    required_volume     = Column(Float, nullable=True)
    volume_ratio        = Column(Float, nullable=True)   # actual / required
    regime              = Column(String(32), nullable=True)
    blocked_reason      = Column(String(64), nullable=True)
    trades_today        = Column(Integer, nullable=True)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ── Public API ────────────────────────────────────────────────────────────────
def init_db():
    """Create all tables if they don't exist. Also runs additive column migrations."""
    Base.metadata.create_all(engine)
    # Additive migrations — safe to run repeatedly (errors = column already exists)
    _migrations = [
        "ALTER TABLE system_status ADD COLUMN session_day INTEGER DEFAULT 1",
    ]
    with engine.connect() as conn:
        for stmt in _migrations:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:
                pass  # column already exists — ignore
    # Seed session day if tables are empty (first boot after PostgreSQL connect)
    _seed_initial_state()
    logger.info("Database initialized.")


def _seed_initial_state():
    """
    On first boot against an empty database, seed system_status and gate_status
    with the session day from INITIAL_SESSION_DAY env var (default 1).
    Safe to call repeatedly — only writes if both tables are empty.
    """
    session_day = int(os.getenv("INITIAL_SESSION_DAY", "1"))
    with Session(engine) as s:
        if s.query(SystemStatus).count() == 0:
            s.add(SystemStatus(
                status="RUNNING",
                regime="Weak Trend",
                trade_count_today=0,
                daily_pnl=0.0,
                consecutive_losses=0,
                kill_switch_active=False,
                session_day=session_day,
                message="Initial state seeded from INITIAL_SESSION_DAY",
            ))
            s.commit()
            logger.info(f"Seeded system_status with session_day={session_day}")
        if s.query(GateStatus).count() == 0:
            s.add(GateStatus(
                day_number=session_day,
                all_passed=False,
                notes=f"Seeded: Day {session_day}/20",
            ))
            s.commit()
            logger.info(f"Seeded gate_status with day_number={session_day}")


def log_trade(data: dict):
    """Insert a completed trade record."""
    with Session(engine) as session:
        trade = Trade(**{k: v for k, v in data.items() if k in Trade.__table__.columns.keys()})
        session.add(trade)
        session.commit()
        logger.info(f"Trade logged: {data.get('symbol')} PnL=${data.get('pnl_dollars', 0):.2f}")


def log_status(data: dict):
    """Insert a system status row."""
    with Session(engine) as session:
        row = SystemStatus(**{k: v for k, v in data.items() if k in SystemStatus.__table__.columns.keys()})
        session.add(row)
        session.commit()


def log_risk_check(check_type: str, result: str, reason: str):
    """Insert a risk check record."""
    with Session(engine) as session:
        row = RiskCheck(check_type=check_type, result=result, reason=reason)
        session.add(row)
        session.commit()


def log_anomaly(data: dict):
    """Insert an anomaly record."""
    with Session(engine) as session:
        ctx = data.get("context_json") or data.get("context")
        if isinstance(ctx, dict):
            ctx = json.dumps(ctx)
        row = Anomaly(
            event_type=data.get("event_type"),
            severity=data.get("severity", "INFO"),
            diagnosis=data.get("diagnosis"),
            recommendation=data.get("recommendation"),
            context_json=ctx,
        )
        session.add(row)
        session.commit()
        logger.warning(f"Anomaly logged: [{data.get('severity')}] {data.get('event_type')}")


def get_today_summary() -> dict:
    """Return aggregated today's data as a dict."""
    today = date.today()
    with Session(engine) as session:
        row = session.get(DailySummary, today)
        if row:
            return row.to_dict()
        # Compute from trades table
        from sqlalchemy import func, cast, Date as SADate
        trades = (
            session.query(Trade)
            .filter(cast(Trade.exit_time, SADate) == today)
            .all()
        )
        if not trades:
            return {}
        wins = [t for t in trades if (t.pnl_dollars or 0) > 0]
        pnls = [t.pnl_dollars or 0 for t in trades]
        rs = [t.pnl_r or 0 for t in trades]
        return {
            "trade_date": today.isoformat(),
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "gross_pnl": sum(pnls),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "avg_r": sum(rs) / len(rs) if rs else 0,
        }


def get_recent_trades(n: int = 50) -> list:
    """Return the N most recent trades as a list of dicts."""
    with Session(engine) as session:
        rows = (
            session.query(Trade)
            .order_by(Trade.id.desc())
            .limit(n)
            .all()
        )
        return [r.to_dict() for r in rows]


def get_gate_status() -> dict:
    """Return the most recent gate_status row."""
    with Session(engine) as session:
        row = session.query(GateStatus).order_by(GateStatus.id.desc()).first()
        return row.to_dict() if row else {}


def get_risk_log(n: int = 20) -> list:
    """Return the N most recent risk check records."""
    with Session(engine) as session:
        rows = (
            session.query(RiskCheck)
            .order_by(RiskCheck.id.desc())
            .limit(n)
            .all()
        )
        return [r.to_dict() for r in rows]


def get_recent_anomalies(n: int = 10) -> list:
    """Return the N most recent anomaly records."""
    with Session(engine) as session:
        rows = (
            session.query(Anomaly)
            .order_by(Anomaly.id.desc())
            .limit(n)
            .all()
        )
        return [r.to_dict() for r in rows]


def get_latest_status() -> dict:
    """Return the most recent system_status row."""
    with Session(engine) as session:
        row = session.query(SystemStatus).order_by(SystemStatus.id.desc()).first()
        return row.to_dict() if row else {}


def log_daily_summary(data: dict):
    """Upsert today's daily summary."""
    today = date.today()
    with Session(engine) as session:
        existing = session.get(DailySummary, today)
        if existing:
            for k, v in data.items():
                if k in DailySummary.__table__.columns.keys():
                    setattr(existing, k, v)
        else:
            row = DailySummary(**{k: v for k, v in data.items() if k in DailySummary.__table__.columns.keys()})
            session.add(row)
        session.commit()


def log_near_miss(data: dict):
    """Insert a near-miss signal record.
    Deduplicates: skips write if a row with same symbol+timestamp (minute-rounded) exists.
    """
    with Session(engine) as session:
        # Round to minute for deduplication
        ts = data.get("timestamp") or datetime.now(timezone.utc)
        ts_minute = ts.replace(second=0, microsecond=0)
        sym = data.get("symbol", "SPY")
        exists = (
            session.query(NearMissSignal)
            .filter(
                NearMissSignal.symbol == sym,
                NearMissSignal.timestamp >= ts_minute,
            )
            .first()
        )
        if exists:
            return  # already logged this bar
        row = NearMissSignal(**{
            k: v for k, v in data.items()
            if k in NearMissSignal.__table__.columns.keys()
        })
        session.add(row)
        session.commit()


def get_recent_near_misses(n: int = 20) -> list:
    """Return the N most recent near-miss signal records."""
    with Session(engine) as session:
        rows = (
            session.query(NearMissSignal)
            .order_by(NearMissSignal.id.desc())
            .limit(n)
            .all()
        )
        return [r.to_dict() for r in rows]


def get_last_near_miss() -> dict:
    """Return the single most recent near-miss record, or {}."""
    with Session(engine) as session:
        row = (
            session.query(NearMissSignal)
            .order_by(NearMissSignal.id.desc())
            .first()
        )
        return row.to_dict() if row else {}


def get_today_near_misses() -> list:
    """Return all near-miss records from today (for email report)."""
    from sqlalchemy import cast, Date as SADate
    today = date.today()
    with Session(engine) as session:
        rows = (
            session.query(NearMissSignal)
            .filter(cast(NearMissSignal.timestamp, SADate) == today)
            .order_by(NearMissSignal.id.desc())
            .all()
        )
        return [r.to_dict() for r in rows]


def get_revenue_summary() -> dict:
    """Aggregate P&L, trade counts, win rate, gates, and system state for /api/revenue."""
    from sqlalchemy import func, cast, Date as SADate
    today      = date.today()
    month_start = today.replace(day=1)

    with Session(engine) as s:
        # ── All-time trades ───────────────────────────────────────────────────
        all_trades = s.query(Trade).all()
        pnls_all   = [t.pnl_dollars or 0.0 for t in all_trades]
        wins_all   = [p for p in pnls_all if p > 0]
        total_pnl       = sum(pnls_all)
        trade_count_total = len(all_trades)
        win_rate          = (len(wins_all) / trade_count_total * 100) if trade_count_total else 0.0

        # ── Today ─────────────────────────────────────────────────────────────
        today_trades = (
            s.query(Trade)
            .filter(cast(Trade.exit_time, SADate) == today)
            .all()
        )
        daily_pnl        = sum(t.pnl_dollars or 0.0 for t in today_trades)
        trade_count_today = len(today_trades)

        # ── Last 7 calendar days ──────────────────────────────────────────────
        from datetime import timedelta
        week_ago = today - timedelta(days=7)
        weekly_trades = (
            s.query(Trade)
            .filter(cast(Trade.exit_time, SADate) > week_ago)
            .all()
        )
        weekly_pnl = sum(t.pnl_dollars or 0.0 for t in weekly_trades)

        # ── This calendar month ───────────────────────────────────────────────
        monthly_trades = (
            s.query(Trade)
            .filter(cast(Trade.exit_time, SADate) >= month_start)
            .all()
        )
        monthly_pnl = sum(t.pnl_dollars or 0.0 for t in monthly_trades)

        # ── Gate status ───────────────────────────────────────────────────────
        gate = s.query(GateStatus).order_by(GateStatus.id.desc()).first()
        g = gate.to_dict() if gate else {}

        def _gv(key, default=0.0):
            v = g.get(key)
            return v if v is not None else default

        gate_progress = {
            "net_return":    {"value": _gv("gate1_return"),      "target": 0.0,    "passed": _gv("gate1_return") > 0},
            "violations":    {"value": int(_gv("gate2_violations", 0)), "target": 3, "passed": _gv("gate2_violations", 99) < 3},
            "max_drawdown":  {"value": _gv("gate3_drawdown"),    "target": -8000,  "passed": _gv("gate3_drawdown", -9999) > -8000},
            "win_rate":      {"value": _gv("gate4_winrate"),     "target": 38.0,   "passed": _gv("gate4_winrate") > 0.38},
            "slippage":      {"value": _gv("gate5_slippage"),    "target": 0.10,   "passed": _gv("gate5_slippage", 99) < 0.10},
        }

        # ── System status ─────────────────────────────────────────────────────
        st = s.query(SystemStatus).order_by(SystemStatus.id.desc()).first()
        sv = st.to_dict() if st else {}

        return {
            "daily_pnl":          round(daily_pnl, 2),
            "weekly_pnl":         round(weekly_pnl, 2),
            "monthly_pnl":        round(monthly_pnl, 2),
            "total_pnl":          round(total_pnl, 2),
            "trade_count_today":  trade_count_today,
            "trade_count_total":  trade_count_total,
            "win_rate":           round(win_rate, 2),
            "gate_day":           sv.get("session_day") or g.get("day_number") or 0,
            "gate_progress":      gate_progress,
            "paper_equity":       round(sv.get("daily_pnl") or 0.0, 2),
            "regime":             sv.get("regime") or "Unknown",
            "kill_switch_active": bool(sv.get("kill_switch_active", False)),
        }


# Auto-initialize on import
init_db()
