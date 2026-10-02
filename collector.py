"""
collector.py - Phase 1 data collector for the Flight Price Tracker.

Pulls cached flight prices from the Travelpayouts (Aviasales) Data API for a
fixed set of routes and saves one snapshot per day. Run it once a day; over
time the daily snapshots form the time series the model trains on.
"""

# ── Section 1: Imports and configuration ─────────────────────────────────────
import csv
import json
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# Anchor every path to this file's folder, so the script works no matter
# which directory it's launched from (Task Scheduler, cron, Docker, etc.).
BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")  # reads variables from the .env file into the environment
API_TOKEN = os.getenv("TRAVELPAYOUTS_TOKEN")

API_URL = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"

# Airport pairs to track. Each pair is collected in BOTH directions.
ROUTE_PAIRS = [
    ("NYC", "TPE"),  # Any New York <-> Taipei Taoyuan
    ("NYC", "STL"),  # Any New York <-> St. Louis
    ("NYC", "YVR"),  # Any New York <-> Vancouver
    ("NYC", "PVG"),  # Any New York <-> Shanghai Pudong
    ("NYC", "KEF"),  # Any New York <-> Reykjavik Keflavik
    ("NYC", "LAX"),  # Any New York <-> Los Angeles
]
# Expand each pair into two one-way routes: (A, B) and (B, A).
ROUTES = [route for a, b in ROUTE_PAIRS for route in ((a, b), (b, a))]

MONTHS_AHEAD = 12  # how many months of departure dates to track
CURRENCY = "usd"

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REQUEST_DELAY_SECONDS = 1.0  # pause between calls to be polite to the API
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("collector")


# ── Section 2: Date helper ───────────────────────────────────────────────────
def upcoming_months(n: int, today: date) -> list[str]:
    """Return the next n months as 'YYYY-MM' strings, starting with this month."""
    months = []
    year, month = today.year, today.month
    for _ in range(n):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


# ── Section 3: API call with retries ─────────────────────────────────────────
def fetch_prices(origin: str, destination: str, month: str) -> list[dict] | None:
    """
    Fetch cached one-way prices for one route and one departure month.
    Returns a list of ticket dicts, or None if the request ultimately failed.
    """
    params = {
        "origin": origin,
        "destination": destination,
        "departure_at": month,
        "one_way": "true",
        "direct": "false",
        "sorting": "price",
        "unique": "false",
        "currency": CURRENCY,
        "limit": 1000,
        "page": 1,
    }
    headers = {"X-Access-Token": API_TOKEN}  # token in a header, not the URL
    route = f"{origin}-{destination} {month}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(API_URL, params=params, headers=headers, timeout=15)
        except requests.RequestException as exc:
            log.warning("%s: network error (%s), attempt %d/%d", route, exc, attempt, MAX_RETRIES)
        else:
            if response.status_code == 200:
                body = response.json()
                if body.get("success"):
                    return body.get("data", [])
                # The API answered but reported a problem; retrying won't fix it.
                log.error("%s: API error: %s", route, body.get("error"))
                return None
            if response.status_code in (429, 500, 502, 503, 504):
                # Rate-limited or temporary server trouble: worth retrying.
                log.warning("%s: HTTP %d, attempt %d/%d", route, response.status_code, attempt, MAX_RETRIES)
            else:
                # 401 (bad token), 400 (bad params), etc.: retrying won't help.
                log.error("%s: HTTP %d, not retrying", route, response.status_code)
                return None
        time.sleep(2 ** attempt)  # exponential backoff: 2s, 4s, 8s

    log.error("%s: giving up after %d attempts", route, MAX_RETRIES)
    return None


# ── Section 4: Normalize raw API rows into model-ready rows ──────────────────
FIELDNAMES = [
    "observed_date",
    "observed_at_utc",
    "origin",
    "destination",
    "origin_airport",
    "destination_airport",
    "departure_date",
    "days_until_departure",
    "price",
    "airline",
    "flight_number",
    "transfers",
    "duration_min",
]


def normalize(raw_rows: list[dict], observed_date: date, observed_at: str) -> list[dict]:
    """Turn raw API tickets into flat rows with the fields the model needs."""
    rows = []
    for r in raw_rows:
        departure_at = r.get("departure_at")
        price = r.get("price")
        if not departure_at or price is None:
            continue  # skip incomplete records instead of crashing
        try:
            departure_date = datetime.fromisoformat(departure_at).date()
        except ValueError:
            log.warning("Unparseable departure_at: %r", departure_at)
            continue

        days_until = (departure_date - observed_date).days
        if days_until < 0:
            continue  # stale cache entry for a flight that already left

        rows.append({
            "observed_date": observed_date.isoformat(),
            "observed_at_utc": observed_at,
            "origin": r.get("origin"),
            "destination": r.get("destination"),
            "origin_airport": r.get("origin_airport"),
            "destination_airport": r.get("destination_airport"),
            "departure_date": departure_date.isoformat(),
            "days_until_departure": days_until,
            "price": price,
            "airline": r.get("airline"),
            "flight_number": r.get("flight_number"),
            "transfers": r.get("transfers"),
            "duration_min": r.get("duration"),
        })
    return rows


# ── Section 5: Save raw and processed data ───────────────────────────────────
def save_raw(responses: dict, observed_date: date) -> Path:
    """Save the untouched API responses so data can be reprocessed later."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{observed_date.isoformat()}.json"
    path.write_text(json.dumps(responses, indent=2))
    return path


def save_processed(rows: list[dict], observed_date: date) -> Path:
    """Save today's normalized rows as one CSV. Re-running today overwrites it."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"prices_{observed_date.isoformat()}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ── Section 6: Main run ──────────────────────────────────────────────────────
def main() -> None:
    if not API_TOKEN:
        raise SystemExit("TRAVELPAYOUTS_TOKEN is not set. Add it to your .env file.")

    observed_date = date.today()
    observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    months = upcoming_months(MONTHS_AHEAD, observed_date)
    log.info("Collecting %d routes x %d months: %s", len(ROUTES), len(months), months)

    raw_responses: dict[str, list] = {}
    all_rows: list[dict] = []
    summary: dict[str, int] = {}

    for origin, destination in ROUTES:
        route_key = f"{origin}-{destination}"
        route_count = 0
        for month in months:
            data = fetch_prices(origin, destination, month)
            time.sleep(REQUEST_DELAY_SECONDS)
            if data is None:
                continue  # one failed call shouldn't stop the whole run
            raw_responses[f"{route_key}_{month}"] = data
            rows = normalize(data, observed_date, observed_at)
            all_rows.extend(rows)
            route_count += len(rows)
        summary[route_key] = route_count

    if not all_rows:
        log.error("Collected 0 rows across all routes; not saving. Check token and API status.")
        raise SystemExit(1)
    
    raw_path = save_raw(raw_responses, observed_date)
    csv_path = save_processed(all_rows, observed_date)

    log.info("Saved raw responses to %s", raw_path)
    log.info("Saved %d rows to %s", len(all_rows), csv_path)
    for route_key, count in summary.items():
        if count == 0:
            log.warning("  %s: 0 rows (little or no cached data for this route)", route_key)
        else:
            log.info("  %s: %d rows", route_key, count)


if __name__ == "__main__":
    main()
