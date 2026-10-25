"""
dataset.py - Turns daily snapshots into a labeled dataset for modeling.

Shared by baseline.py and (later) the PyTorch model, so both are trained and
tested on exactly the same examples.

One example = one route + departure date, as seen on one observation day.
Label = 1 if the cheapest price is higher HORIZON_DAYS later, else 0.
"""

import glob
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

HORIZON_DAYS = 7
MIN_CHANGE = 0.5  # dollars; smaller differences count as "no change"

ROUTE_PAIRS = [
    ("NYC", "TPE"),
    ("NYC", "STL"),
    ("NYC", "YVR"),
    ("NYC", "SHA"),
    ("NYC", "REK"),
    ("NYC", "LAX"),
]
CURRENT_ROUTES = sorted(
    {f"{a}-{b}" for a, b in ROUTE_PAIRS} | {f"{b}-{a}" for a, b in ROUTE_PAIRS}
)
CITY_ALIASES = {"SWF": "NYC", "ONT": "LAX", "LGB": "LAX"}


# ── Section 1: Load and clean snapshots ──────────────────────────────────────
def load_snapshots() -> pd.DataFrame:
    """Load every daily CSV, label routes, and drop early test routes."""
    files = sorted(glob.glob(str(PROCESSED_DIR / "*.csv")))
    if not files:
        raise SystemExit(f"No snapshot files found in {PROCESSED_DIR}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["route"] = (
        df.origin.replace(CITY_ALIASES) + "-" + df.destination.replace(CITY_ALIASES)
    )
    df = df[df.route.isin(CURRENT_ROUTES)].copy()
    df["observed_date"] = pd.to_datetime(df.observed_date)
    df["departure_date"] = pd.to_datetime(df.departure_date)
    return df


# ── Section 2: One cheapest price per route, departure date, and day ─────────
def cheapest_series(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["route", "departure_date", "observed_date"])
        .price.min()
        .reset_index()
    )


def _price_n_days_away(series: pd.DataFrame, offset_days: int, name: str) -> pd.DataFrame:
    """For each row, look up the same series' price offset_days away (or NaN)."""
    shifted = series[["route", "departure_date", "observed_date", "price"]].copy()
    shifted["observed_date"] = shifted.observed_date - pd.Timedelta(days=offset_days)
    shifted = shifted.rename(columns={"price": name})
    return series.merge(shifted, on=["route", "departure_date", "observed_date"], how="left")


# ── Section 3: Features and label ────────────────────────────────────────────
def build_dataset() -> pd.DataFrame:
    """Return one row per labeled example, with features and a 0/1 label."""
    s = cheapest_series(load_snapshots())

    # Label: the price HORIZON_DAYS in the future. Only known for some rows.
    s = _price_n_days_away(s, HORIZON_DAYS, "future_price")
    # Feature: the price 3 days in the past (recent trend), if observed.
    s = _price_n_days_away(s, -3, "price_3d_ago")

    s["days_until_departure"] = (s.departure_date - s.observed_date).dt.days
    s["departure_dow"] = s.departure_date.dt.dayofweek  # 0 = Monday
    s["recent_change"] = (s.price - s.price_3d_ago).fillna(0.0)
    s["has_recent_price"] = s.price_3d_ago.notna().astype(int)
    # Price relative to the route's typical level, so routes are comparable.
    s["price_vs_route_median"] = s.price / s.groupby("route").price.transform("median")

    # Keep only rows whose future price was actually observed, and whose
    # flight hasn't departed by the label date.
    s = s[s.future_price.notna() & (s.days_until_departure > HORIZON_DAYS)].copy()
    s["label"] = (s.future_price - s.price > MIN_CHANGE).astype(int)
    return s.reset_index(drop=True)


# ── Section 4: Time-based train/test split ───────────────────────────────────
def time_split(data: pd.DataFrame, test_days: int = 5):
    """
    Train on earlier observation days, test on the last `test_days` days.

    A training row's label is only known HORIZON_DAYS after it was observed.
    So training rows must be observed at least HORIZON_DAYS before the test
    period starts; otherwise their labels would "peek" into test-period prices.
    """
    obs_days = sorted(data.observed_date.unique())
    if len(obs_days) <= test_days:
        raise SystemExit("Not enough labeled days to split. Collect more data.")
    test_start = obs_days[-test_days]
    train_cutoff = test_start - pd.Timedelta(days=HORIZON_DAYS)

    train = data[data.observed_date <= train_cutoff]
    test = data[data.observed_date >= test_start]
    return train, test
