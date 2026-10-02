# Flight Price Tracker & Predictor

An end-to-end system that collects flight price data over time and predicts future price movement, built to demonstrate cloud infrastructure (AWS), containerization (Docker, Kubernetes), and deep learning (PyTorch/TensorFlow) working together as a real pipeline, not just as isolated scripts.

## Project Status

Started: September 2026

- [ ] Phase 1: Data collector + baseline model
  - [x] Data collector
  - [x] Automated daily collection with GitHub Actions
  - [ ] Data exploration and validation
  - [ ] Baseline model, then PyTorch/TensorFlow model
- [ ] Phase 2: Containerization (Docker)
- [ ] Phase 3: Cloud deployment (AWS)
- [ ] Phase 4 (stretch): Kubernetes orchestration

## Why This Project

Flight price history isn't freely available: prices are generated per search, and nobody publishes a free archive of them. So this project builds its own growing dataset by collecting real flight prices on a schedule, then uses that data to train a model that predicts whether prices on a given route are likely to rise or fall. The point of the project is the full pipeline, automated collection, containerized services, cloud deployment, and a deep learning model working together, not just the model in isolation.

## Phase 1: What's Built So Far

### Data collector (`collector.py`)

Each run requests one-way prices for every tracked route, in both directions, for each of the next 12 departure months. It cleans the results into model-ready rows and saves two files per day:

- `data/raw/YYYY-MM-DD.json`: the untouched API responses
- `data/processed/prices_YYYY-MM-DD.csv`: cleaned rows, one per flight price observed

Key design choices:

- **Retries that distinguish error types.** Rate limits (429) and server errors (5xx) are retried with exponential backoff. Permanent errors like a bad token (401) fail immediately, since retrying can't fix them.
- **Graceful partial failure.** If one route or month fails, it's logged and skipped while the rest of the run continues.
- **Loud failure on empty runs.** If a run collects zero rows in total, the script exits with an error code instead of saving empty files, so the failure is flagged rather than silently ignored.
- **Raw data preserved.** Past prices can't be re-fetched, so keeping raw responses means the data can be reprocessed if the cleaning logic changes.
- **Idempotent daily files.** Re-running on the same day overwrites that day's files instead of creating duplicates.
- **Secrets kept out of code.** The API token comes from an environment variable, never from the source.

### Automated collection with GitHub Actions

Collection runs in the cloud on a daily schedule, so the dataset grows whether or not my laptop is on.

```
GitHub Actions (daily cron, 00:21 UTC)
        │
        ▼
collector.py ──► Travelpayouts Data API
        │
        ▼
data/raw/ + data/processed/
        │
        ▼
Committed back to this repo by the workflow
```

The workflow (`.github/workflows/collect.yml`):

- Runs on a cron schedule, with a manual "Run workflow" button for testing
- Injects the API token from an encrypted GitHub Actions secret
- Installs only the collector's two dependencies, not the full ML stack, to keep runs fast
- Commits each day's snapshot back to the repository, skipping the commit if nothing changed
- Fails visibly (with an email alert) if the collector exits with an error

This is an interim setup: in Phase 3, scheduled collection moves to AWS.

### Routes tracked

Each route is collected in both directions.

- New York ↔ Taipei (TPE)
- Newark ↔ St. Louis (STL)
- Newark ↔ Vancouver (YVR)
- New York ↔ Shanghai Pudong (PVG)
- Newark ↔ Reykjavik Keflavik (KEF)
- New York ↔ Los Angeles (LAX)

"New York" uses the city code `NYC`, which covers JFK, Newark, and LaGuardia. The exact airport for each flight is recorded in the data.

## Data

### Schema (`data/processed/prices_YYYY-MM-DD.csv`)

| Column | Description |
|---|---|
| `observed_date` | Date the price was recorded (UTC). Turns daily snapshots into a time series. |
| `observed_at_utc` | Exact timestamp of the run |
| `origin`, `destination` | Codes as requested (city or airport) |
| `origin_airport`, `destination_airport` | Exact airports for the flight |
| `departure_date` | Flight departure date |
| `days_until_departure` | Departure date minus observed date |
| `price` | Price in USD |
| `airline`, `flight_number` | Carrier details |
| `transfers` | Number of stops |
| `duration_min` | Total trip duration in minutes |

### Source and limitations

- **Source.** Prices come from the Travelpayouts (Aviasales) Data API. The project was originally planned around the Amadeus Self-Service API, which Amadeus shut down in July 2026.
- **Cached prices.** The API returns prices from searches made by Aviasales users in roughly the last 48 hours, not live quotes, so a price can repeat across consecutive days. Collected prices will be spot-checked against Google Flights price history.
- **Uneven coverage.** Less-searched routes and dates return fewer prices, so some routes have gaps.
- **One holiday season so far.** Collection began September 27, 2026, so the data captures the 2026 holiday price run-up once. Confirming the pattern repeats requires a second season.
- **Date convention.** Dates are in UTC. The earliest snapshots (Sept 27 to Oct 1) were collected locally using Central time; the Oct 1 local snapshot and Oct 2 Actions snapshot were taken minutes apart and count as one observation.

## Running It Yourself

**Requirements:** Python 3.10+ and a free [Travelpayouts](https://www.travelpayouts.com) account (API token under Profile → API token).

```bash
git clone <this-repo-url>
cd flight-price-tracker
python -m pip install requests python-dotenv
```

Copy `.env.example` to `.env` and add your token:

```
TRAVELPAYOUTS_TOKEN=your_token_here
```

Run one collection:

```bash
python collector.py
```

**To schedule it with GitHub Actions:** add your token as a repository secret named `TRAVELPAYOUTS_TOKEN` (Settings → Secrets and variables → Actions). The workflow then runs daily and can also be started manually from the Actions tab.

## Planned Architecture

```
[Scheduled Data Collector] --> [S3: raw price data]
                                       |
                                       v
                              [Training Job: PyTorch/TensorFlow]
                                       |
                                       v
                              [Trained Model: S3]
                                       |
                                       v
                        [Inference API] <-- [User/request]
```

All components containerized with Docker. Deployed on AWS (S3, EventBridge/Lambda or ECS for scheduling, API Gateway or ECS for serving predictions). Kubernetes orchestration is a stretch goal once the core pipeline is stable.

## Phase Breakdown

### Phase 1: Data Collector + Baseline Model
- Pull flight price data on a schedule for a fixed set of routes (done, see above)
- Store collected data as the dataset grows (done, committed daily by GitHub Actions)
- Build a baseline time series model (starting simple, then a PyTorch/TensorFlow model) to predict price direction

### Phase 2: Containerization
- Split the system into separate services: data collector, model training, inference API
- Write a Dockerfile for each service
- Connect them locally with docker-compose
- Confirm the full pipeline runs end to end in containers before touching the cloud

### Phase 3: AWS Deployment
- S3 for storing raw data and trained models
- A scheduled trigger (EventBridge + Lambda, or a container on ECS) running the data collector automatically, replacing GitHub Actions
- An inference endpoint (API Gateway + Lambda, or ECS service) serving predictions
- IAM roles scoped correctly for each service

### Phase 4 (Stretch): Kubernetes
- Migrate the containerized services to run under Kubernetes instead of directly on ECS
- Primarily a learning exercise in orchestration, not a requirement for this project's actual scale

## Tech Stack

**In use:** Python, Requests, Travelpayouts Data API, GitHub Actions

**Planned:** Pandas, PyTorch or TensorFlow, Docker, Docker Compose, AWS (S3, Lambda, EventBridge, API Gateway/ECS), Kubernetes (stretch)

## Project Structure

```
flight-price-tracker/
├── .github/workflows/collect.yml   # daily scheduled collection
├── collector.py                    # data collector
├── data/
│   ├── raw/                        # daily raw API responses (JSON)
│   └── processed/                  # daily cleaned snapshots (CSV)
├── .env.example                    # template for the API token
├── requirements.txt
└── README.md
```
