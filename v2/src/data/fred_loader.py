"""FRED Economic Data Loader for QMKL-v2.

Builds a financial classification dataset from FRED (Federal Reserve Economic Data):
- Features  : ~20 macroeconomic indicators (unemployment, rates, spreads, production...)
- Target    : USREC — NBER recession indicator (0=expansion, 1=recession)
- Frequency : monthly, 1970-present
- Samples   : ~600 months after alignment and lag

This is the primary "real financial" dataset for QMKL-v2.
It has d=20 features >> Q=4 qubits, justifying the QUBO-assignation approach.

Requires FRED_API_KEY environment variable (free at https://fred.stlouisfed.org/).

Usage:
    X, y, feature_names = load_fred_recession_data()
    # or with explicit key:
    X, y, feature_names = load_fred_recession_data(api_key="your_key")
"""

import os
import sys
import numpy as np
import pandas as pd

# Charge automatiquement .env depuis la racine du projet
def _load_dotenv():
    """Parse .env sans dépendance externe."""
    env_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    )
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val

_load_dotenv()

# Allow using the skill's FREDQuery class
_SKILL_PATH = os.path.expanduser("~/.claude/skills/fred-economic-data/scripts")
if _SKILL_PATH not in sys.path:
    sys.path.insert(0, _SKILL_PATH)


# ── FRED series used as features ──────────────────────────────────────────────
#
# Chosen to cover 5 economic dimensions:
#   1. Labor market      : unemployment, payrolls
#   2. Monetary/rates    : Fed Funds, Treasury yields, yield spread
#   3. Credit/risk       : High Yield spread, Baa spread, TED spread
#   4. Real activity     : Industrial production, retail sales, housing
#   5. Inflation/money   : CPI, M2, oil price, consumer sentiment

FRED_FEATURES = {
    # Labor market
    "UNRATE":       ("Unemployment Rate (%)",              "lin"),
    "PAYEMS":       ("Nonfarm Payrolls (YoY %chg)",        "pc1"),

    # Monetary policy & rates
    "FEDFUNDS":     ("Federal Funds Rate (%)",             "lin"),
    "DGS10":        ("10Y Treasury Yield (%)",             "lin"),
    "DGS2":         ("2Y Treasury Yield (%)",              "lin"),
    "T10Y2Y":       ("10Y-2Y Yield Spread (bp)",           "lin"),
    "T10Y3M":       ("10Y-3M Yield Spread (bp)",           "lin"),

    # Credit & risk premia
    "BAMLH0A0HYM2": ("High Yield Spread (bp)",             "lin"),
    "BAA10Y":       ("Baa-Treasury Spread (bp)",           "lin"),
    "TEDRATE":      ("TED Spread (bp)",                    "lin"),

    # Real activity
    "INDPRO":       ("Industrial Production (YoY %chg)",   "pc1"),
    "RSAFS":        ("Retail Sales (YoY %chg)",            "pc1"),
    "HOUST":        ("Housing Starts (YoY %chg)",          "pc1"),
    "PERMIT":       ("Building Permits (YoY %chg)",        "pc1"),

    # Inflation & money
    "CPIAUCSL":     ("CPI Inflation (YoY %chg)",           "pc1"),
    "M2SL":         ("M2 Money Stock (YoY %chg)",          "pc1"),
    "DCOILWTICO":   ("Oil Price WTI (YoY %chg)",           "pc1"),

    # Sentiment & market
    "UMCSENT":      ("Consumer Sentiment",                 "lin"),
    "SP500":        ("S&P500 (YoY %chg)",                  "pc1"),
}

# Binary target: NBER recession indicator (0=expansion, 1=recession)
TARGET_SERIES = "USREC"

# Date range — enough recessions: 1970 covers 9 NBER recessions
DEFAULT_START = "1970-01-01"
DEFAULT_END   = "2023-12-01"


# ── Main loader ───────────────────────────────────────────────────────────────

def load_fred_recession_data(
    api_key=None,
    start_date=DEFAULT_START,
    end_date=DEFAULT_END,
    lag_months=1,
    cache_path=None,
):
    """Load FRED macroeconomic classification dataset (recession vs expansion).

    Fetches 20 FRED time series + USREC target, aligns to monthly frequency,
    handles missing values, and applies a lag to avoid look-ahead bias.

    Args:
        api_key    : FRED API key (uses FRED_API_KEY env var if None)
        start_date : Observation start (YYYY-MM-DD)
        end_date   : Observation end (YYYY-MM-DD)
        lag_months : Lag features by N months to avoid data leakage (default=1)
        cache_path : Optional path to cache the dataset as CSV

    Returns:
        X            : (N, d) float array of macro features
        y            : (N,) int array, 1=recession, 0=expansion
        feature_names: list of d feature names (human-readable)
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")

    # Try loading from cache first
    if cache_path and os.path.exists(cache_path):
        return _load_from_cache(cache_path)

    if not api_key:
        raise ValueError(
            "FRED API key required.\n"
            "  1. Get a free key at https://fred.stlouisfed.org/\n"
            "  2. Set: export FRED_API_KEY=<your_key>\n"
            "  Or pass: load_fred_recession_data(api_key='your_key')"
        )

    try:
        from fred_query import FREDQuery
    except ImportError:
        raise ImportError(
            "FREDQuery not found. The fred-economic-data skill must be installed.\n"
            f"Expected at: {_SKILL_PATH}/fred_query.py"
        )

    fred = FREDQuery(api_key=api_key)
    print(f"Fetching {len(FRED_FEATURES)} FRED series + recession indicator...")

    # ── Fetch all series ───────────────────────────────────────────────────
    series_data = {}

    for series_id, (label, units) in FRED_FEATURES.items():
        print(f"  → {series_id:20s} ({label})", end=" ")
        obs = fred.get_observations(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
            units=units,
            frequency="m",
            aggregation_method="avg",
        )
        if "error" in obs:
            print(f"[SKIP — {obs['error']['message']}]")
            continue

        values = {
            o["date"]: float(o["value"])
            for o in obs["observations"]
            if o["value"] not in (".", "", None)
        }
        series_data[series_id] = values
        print(f"[{len(values)} obs]")

    # ── Fetch recession target ─────────────────────────────────────────────
    print(f"  → {TARGET_SERIES:20s} (NBER Recession Indicator)", end=" ")
    obs_rec = fred.get_observations(
        TARGET_SERIES,
        observation_start=start_date,
        observation_end=end_date,
        units="lin",
        frequency="m",
    )
    recession = {
        o["date"]: int(float(o["value"]))
        for o in obs_rec["observations"]
        if o["value"] not in (".", "", None)
    }
    print(f"[{len(recession)} obs — {sum(recession.values())} recession months]")

    # ── Align into DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame(series_data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Apply lag to avoid look-ahead bias
    if lag_months > 0:
        df = df.shift(lag_months)

    # Add target (not lagged)
    df_rec = pd.Series(recession)
    df_rec.index = pd.to_datetime(df_rec.index)
    df["USREC"] = df_rec

    # Drop rows with any NaN
    df = df.dropna()

    # ── Extract X, y ───────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != "USREC"]
    X = df[feature_cols].values.astype(float)
    y = df["USREC"].values.astype(int)

    feature_names = [FRED_FEATURES[c][0] for c in feature_cols]

    # Cache if requested
    if cache_path:
        df.to_csv(cache_path)
        print(f"\nDataset cached to: {cache_path}")

    print(f"\nDataset FRED Recession:")
    print(f"  Samples   : {len(y)} monthly observations")
    print(f"  Features  : {X.shape[1]} macroeconomic indicators")
    print(f"  Target    : {y.sum()} recession / {(y==0).sum()} expansion months")
    print(f"  Class ratio: {y.mean():.1%} recession")

    return X, y, feature_names


def _load_from_cache(cache_path):
    """Load dataset from cached CSV."""
    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    feature_cols = [c for c in df.columns if c != "USREC"]
    X = df[feature_cols].values.astype(float)
    y = df["USREC"].values.astype(int)
    # Reconstruct feature names from column IDs
    feature_names = [
        FRED_FEATURES[c][0] if c in FRED_FEATURES else c
        for c in feature_cols
    ]
    print(f"Dataset chargé depuis cache ({cache_path}) : {X.shape}")
    return X, y, feature_names


# ── Convenience: offline synthetic fallback ────────────────────────────────────

def load_fred_recession_synthetic(n_samples=600, seed=42):
    """Synthetic fallback mimicking FRED recession dataset structure.

    Use when FRED_API_KEY is not available (tests, CI, demos).
    Generates realistic class imbalance (~15% recession) and
    feature correlations inspired by actual macro relationships.

    Returns:
        X            : (n_samples, 19) float array
        y            : (n_samples,) int array
        feature_names: list of feature names
    """
    rng = np.random.RandomState(seed)
    feature_names = [v[0] for v in FRED_FEATURES.values()]
    d = len(feature_names)

    # Latent recession factor
    recession_prob = 0.15
    y = (rng.rand(n_samples) < recession_prob).astype(int)

    # Features with recession signal
    X = rng.randn(n_samples, d)

    # Add recession signal to relevant features
    recession_signals = {
        0: +2.0,   # Unemployment rises
        1: -3.0,   # Payrolls fall
        5: -1.5,   # Yield spread inverts
        6: -2.0,   # 10Y-3M inverts strongly
        7: +2.5,   # HY spread widens
        8: +1.5,   # Baa spread widens
        10: -2.0,  # Industrial production falls
        11: -1.5,  # Retail sales fall
        14: -1.0,  # CPI drops (deflation risk)
        18: -3.0,  # S&P500 falls
    }
    rec_idx = y == 1
    for feat_idx, signal in recession_signals.items():
        X[rec_idx, feat_idx] += signal

    print(f"Synthetic FRED dataset : {X.shape}, {y.sum()} recession / {(y==0).sum()} expansion")
    return X, y, feature_names
