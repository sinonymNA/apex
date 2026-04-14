"""
models/regime_classifier.py — Market regime classification.

If models/regime_rf.pkl exists: load and use it.
If not: run in default mode (always return "Weak Trend").

Regimes: Strong Trend, Weak Trend, Range-Bound,
         High Volatility, Extreme Volatility

Train with: RegimeClassifier().train_and_save(daily_df)
Called automatically by backtest/run.py.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_MODEL_PATH = Path(__file__).parent / "regime_rf.pkl"

# Lazy imports for heavy dependencies
_joblib = None
_GaussianHMM = None
_RandomForestClassifier = None
_StandardScaler = None


def _import_ml_deps():
    global _joblib, _GaussianHMM, _RandomForestClassifier, _StandardScaler
    if _joblib is None:
        import joblib as _jl
        from hmmlearn.hmm import GaussianHMM as _HMM
        from sklearn.ensemble import RandomForestClassifier as _RFC
        from sklearn.preprocessing import StandardScaler as _SS
        _joblib = _jl
        _GaussianHMM = _HMM
        _RandomForestClassifier = _RFC
        _StandardScaler = _SS


# ── Regime semantic labels ─────────────────────────────────────────────────────
REGIME_LABELS = [
    "Strong Trend",
    "Weak Trend",
    "Range-Bound",
    "High Volatility",
    "Extreme Volatility",
]


class RegimeClassifier:
    """
    Market regime classifier backed by HMM-labeled training data and
    a Random Forest classifier on rolling features.

    Thread-safe: model is loaded once at init; classify() is read-only.
    """

    def __init__(self, model_path: Path = _MODEL_PATH):
        self.model_path = model_path
        self.rf_model = None
        self.scaler = None
        self.state_map = {}
        self._load_or_default()

    def _load_or_default(self):
        if self.model_path.exists():
            try:
                _import_ml_deps()
                bundle = _joblib.load(self.model_path)
                self.rf_model = bundle["model"]
                self.scaler = bundle["scaler"]
                self.state_map = bundle["state_map"]
                logger.info(f"Regime model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load regime model: {e} — defaulting to 'Weak Trend'")
                self.rf_model = None
        else:
            logger.info("No regime model found — defaulting to 'Weak Trend'")

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a 5-feature matrix from an OHLCV DataFrame.
        Expects uppercase column names (Close, High, Low, Volume).
        """
        df = df.copy()
        close = df["Close"]

        returns = close.pct_change()
        vol_20 = returns.rolling(20).std()
        mean_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        trend_z = (close - mean_20) / std_20.replace(0, np.nan)

        # ATR ratio (manual, no ta dependency here)
        prev_close = close.shift(1)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atr_ratio = atr14 / close.replace(0, np.nan)

        vol_ratio = df["Volume"] / df["Volume"].rolling(20).mean().replace(0, np.nan)

        features = pd.DataFrame({
            "returns": returns,
            "vol_20": vol_20,
            "trend_z": trend_z,
            "atr_ratio": atr_ratio,
            "vol_ratio": vol_ratio,
        }, index=df.index)

        return features

    def classify(self, df: pd.DataFrame) -> str:
        """
        Classify the current market regime from OHLCV data.

        Args:
            df: DataFrame with at least 30 recent bars (Close, High, Low, Volume)

        Returns:
            One of: Strong Trend, Weak Trend, Range-Bound,
                    High Volatility, Extreme Volatility
        """
        if self.rf_model is None:
            return "Weak Trend"

        try:
            feats = self._build_features(df).dropna()
            if feats.empty:
                return "Weak Trend"

            last_row = feats.iloc[-1:].values
            scaled = self.scaler.transform(last_row)
            raw_state = int(self.rf_model.predict(scaled)[0])
            return self.state_map.get(raw_state, "Weak Trend")
        except Exception as e:
            logger.error(f"Regime classification error: {e}")
            return "Weak Trend"

    def train_and_save(self, df: pd.DataFrame):
        """
        Train the HMM + Random Forest regime classifier and save the pkl.

        Args:
            df: Daily OHLCV DataFrame (5+ years recommended) with uppercase columns.

        The HMM finds 5 hidden states. States are semantically labeled by ranking
        on (volatility, mean_return):
          - Lowest std + positive mean → Strong Trend
          - Lowest std + near-zero mean → Range-Bound
          - Medium std + positive mean → Weak Trend
          - High std → High Volatility
          - Highest std → Extreme Volatility

        The state_map dict (hmm_state_int → regime_str) is saved alongside the model
        so classify() always returns consistent labels across restarts.
        """
        _import_ml_deps()
        logger.info("Training regime classifier...")

        # Build features and HMM input — clean NaN and Inf
        returns = df["Close"].pct_change().dropna()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < 50:
            raise ValueError(f"Too few clean return observations ({len(returns)}) to train HMM")
        hmm_input = returns.values.reshape(-1, 1)

        # Fit 5-state Gaussian HMM — use "diag" covariance for numerical stability
        hmm = _GaussianHMM(
            n_components=5,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
            verbose=False,
            min_covar=1e-5,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm.fit(hmm_input)
        hmm_states = hmm.predict(hmm_input)

        # Verify no NaN in states
        if np.any(np.isnan(hmm_states.astype(float))):
            raise ValueError("HMM predicted NaN states — training data may be too homogeneous")

        # ── Semantic state mapping ─────────────────────────────────────────────
        # For each HMM state, compute mean return and std of returns
        state_stats = {}
        for s in range(5):
            mask = hmm_states == s
            if mask.sum() == 0:
                state_stats[s] = {"mean": 0.0, "std": 0.01}
                continue
            state_returns = returns.values[mask]
            state_stats[s] = {
                "mean": float(np.mean(state_returns)),
                "std": float(np.std(state_returns)),
            }

        # Sort states by volatility (std) ascending
        sorted_by_vol = sorted(state_stats.keys(), key=lambda s: state_stats[s]["std"])

        # Assign labels based on volatility rank and mean return
        state_map = {}
        low_vol_states = sorted_by_vol[:2]
        high_vol_states = sorted_by_vol[3:]
        mid_vol_states = sorted_by_vol[2:3]

        # Among low-vol states, higher mean return → Strong Trend, else Range-Bound
        low_vol_sorted_by_mean = sorted(low_vol_states, key=lambda s: state_stats[s]["mean"], reverse=True)
        state_map[low_vol_sorted_by_mean[0]] = "Strong Trend"
        state_map[low_vol_sorted_by_mean[1]] = "Range-Bound"

        # Mid-vol state → Weak Trend
        for s in mid_vol_states:
            state_map[s] = "Weak Trend"

        # High-vol states: highest → Extreme Volatility
        high_vol_sorted = sorted(high_vol_states, key=lambda s: state_stats[s]["std"], reverse=True)
        state_map[high_vol_sorted[0]] = "Extreme Volatility"
        if len(high_vol_sorted) > 1:
            state_map[high_vol_sorted[1]] = "High Volatility"

        logger.info(f"HMM state mapping: {state_map}")
        logger.info(f"State stats: {state_stats}")

        # ── Build RF training data ─────────────────────────────────────────────
        # Align features with HMM labels (HMM labels start from returns index)
        features_df = self._build_features(df)
        returns_index = returns.index  # index after pct_change().dropna()

        # HMM states align with returns_index
        labels_series = pd.Series(hmm_states, index=returns_index)

        # Align features and labels — drop NaN and Inf rows
        aligned_features = features_df.loc[returns_index].replace([np.inf, -np.inf], np.nan).dropna()
        aligned_labels = labels_series.loc[aligned_features.index]

        X = aligned_features.values
        y = aligned_labels.values

        if len(X) < 20:
            raise ValueError(f"Too few clean training samples ({len(X)}) for RF classifier")

        logger.info(f"Training RF on {len(X)} samples...")

        # Scale and train
        scaler = _StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = _RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)

        # ── Save bundle ────────────────────────────────────────────────────────
        bundle = {
            "model": rf,
            "scaler": scaler,
            "state_map": state_map,
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        _joblib.dump(bundle, self.model_path)
        logger.info(f"Regime model saved to {self.model_path}")

        # Update instance
        self.rf_model = rf
        self.scaler = scaler
        self.state_map = state_map

        return state_map


# Module-level singleton
classifier = RegimeClassifier()
