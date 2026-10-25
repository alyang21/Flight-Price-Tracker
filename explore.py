"""
explore.py - Data exploration for the Flight Price Tracker.

Loads every daily snapshot and checks whether the data can support a model:
coverage, gaps, and above all whether prices actually move between snapshots.
"""

import glob

import pandas as pd

# Routes currently collected (each tracked in both directions).
ROUTE_PAIRS = [
    ("NYC", "TPE"),
    ("NYC", "STL"),
    ("NYC", "YVR"),
    ("NYC", "SHA"),  # requested as PVG; the API labels Shanghai with its city code
    ("NYC", "REK"),  # requested as KEF; the API labels Reykjavik with its city code
    ("NYC", "LAX"),
]

# The API sometimes labels nearby airports with their own city codes.
# Map them back to the route they were collected under.
CITY_ALIASES = {
    "SWF": "NYC",  # Newburgh/Stewart, returned for New York requests
    "ONT": "LAX",  # Ontario, CA, returned for Los Angeles requests
    "LGB": "LAX",  # Long Beach, returned for Los Angeles requests
}

pd.set_option("display.width", 120)

# ── Section 1: Load all snapshots ────────────────────────────────────────────
files = sorted(glob.glob("data/processed/*.csv"))
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
print(f"Loaded {len(df):,} rows from {len(files)} files\n")

# ── Section 2: Snapshot coverage and missing days ────────────────────────────
days = pd.to_datetime(sorted(df.observed_date.unique()))
missing = pd.date_range(days.min(), days.max()).difference(days)
print(f"Snapshot days: {len(days)} ({days.min().date()} to {days.max().date()})")
print("Missing days:", [d.date().isoformat() for d in missing] or "none")
print()

# ── Section 3: Label rows by the route they were collected under ─────────────
df["route"] = (
    df.origin.replace(CITY_ALIASES) + "-" + df.destination.replace(CITY_ALIASES)
)
current_routes = {f"{a}-{b}" for a, b in ROUTE_PAIRS} | {f"{b}-{a}" for a, b in ROUTE_PAIRS}
early = ~df.route.isin(current_routes)
print(f"Dropping {early.sum():,} rows from early test routes")
df = df[~early].copy()

print("\nRows per route:")
print(df.groupby("route").size().sort_values(ascending=False).to_string())
print()

# ── Section 4: Individual flights across days ────────────────────────────────
# A specific flight = same route, airports, departure date, airline, flight number.
flight_key = [
    "route", "origin_airport", "destination_airport",
    "departure_date", "airline", "flight_number",
]
# If a flight appears more than once in a day, keep that day's lowest price.
daily = (
    df.groupby(flight_key + ["observed_date"], dropna=False)
    .price.min()
    .reset_index()
)
per_flight = daily.groupby(flight_key, dropna=False).agg(
    days_seen=("observed_date", "nunique"),
    distinct_prices=("price", "nunique"),
)
repeat = per_flight[per_flight.days_seen >= 2]

print(f"Distinct flights tracked: {len(per_flight):,}")
print(f"  seen on only 1 day:  {(per_flight.days_seen == 1).mean():.0%}")
print(f"  seen on 5+ days:     {(per_flight.days_seen >= 5).mean():.0%}")
print(
    "Of flights seen on 2+ days, share whose price ever changed: "
    f"{(repeat.distinct_prices > 1).mean():.0%}"
)
print()

# ── Section 5: Cheapest price per route and departure date ───────────────────
# The question a traveler asks: "What's the cheapest way to fly this route on
# this date, and will that go up or down?" Pooling across flights gives a
# steadier series than tracking one flight number.
cheapest = (
    df.groupby(["route", "departure_date", "observed_date"])
    .price.min()
    .reset_index()
    .sort_values(["route", "departure_date", "observed_date"])
)
cheapest["prev_price"] = cheapest.groupby(["route", "departure_date"]).price.shift()
moves = cheapest.dropna(subset=["prev_price"])
direction = pd.cut(
    moves.price - moves.prev_price,
    bins=[-float("inf"), -0.5, 0.5, float("inf")],
    labels=["down", "same", "up"],
)

print(f"Route + departure-date series: {cheapest.groupby(['route', 'departure_date']).ngroups:,}")
print(f"Consecutive observation pairs: {len(moves):,}")
print("Price movement between consecutive observations:")
print(direction.value_counts(normalize=True).reindex(["up", "same", "down"]).round(2).to_string())

print("\nSeries with 5+ observations, per route:")
obs_per_series = cheapest.groupby(["route", "departure_date"]).size()
print((obs_per_series >= 5).groupby(level="route").sum().sort_values(ascending=False).to_string())

# ── Section 6: Movement over longer horizons ─────────────────────────────────
# Day-to-day prices are mostly flat, so check how often the cheapest price has
# moved by N days later. This decides the prediction horizon for the model.
cheapest["observed_date"] = pd.to_datetime(cheapest["observed_date"])
print("\nPrice movement after N days (same route and departure date):")
for horizon in (3, 7, 14):
    later = cheapest[["route", "departure_date", "observed_date", "price"]].copy()
    later["observed_date"] = later["observed_date"] - pd.Timedelta(days=horizon)
    pairs = cheapest.merge(
        later, on=["route", "departure_date", "observed_date"], suffixes=("", "_later")
    )
    change = pairs.price_later - pairs.price
    up = (change > 0.5).mean()
    down = (change < -0.5).mean()
    print(
        f"  {horizon:>2} days: {len(pairs):>5,} pairs | "
        f"up {up:.0%}  same {1 - up - down:.0%}  down {down:.0%}"
    )
