# Flight Price Tracker & Predictor

An end-to-end system that collects flight price data over time and predicts future price movement, built to demonstrate cloud infrastructure (AWS), containerization (Docker, Kubernetes), and deep learning (PyTorch/TensorFlow) working together as a real pipeline, not just as isolated scripts.

## Project Status

Started: September 2026

- [x] Phase 1: Data collector + baseline model
  - [x] Data collector
  - [x] Automated daily collection with GitHub Actions
  - [x] Data exploration
  - [x] Baseline models and PyTorch model
- [x] Phase 2: Containerization (Docker)
- [ ] Phase 3: Cloud deployment (AWS)
- [ ] Phase 4 (stretch): Kubernetes orchestration

## Why This Project

Flight price history isn't freely available: prices are generated per search, and nobody publishes a free archive of them. So this project builds its own growing dataset by collecting real flight prices on a schedule, then uses that data to train a model that predicts whether prices on a given route are likely to rise or fall. The point of the project is the full pipeline, automated collection, containerized services, cloud deployment, and a deep learning model working together, not just the model in isolation.

## Phase 1: Data Collection

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

Since automation began on October 1, 2026, no daily snapshot has been missed. This is an interim setup: in Phase 3, scheduled collection moves to AWS.

### Routes tracked

Each route is collected in both directions.

- New York ↔ Taipei (TPE)
- New York ↔ St. Louis (STL)
- New York ↔ Vancouver (YVR)
- New York ↔ Shanghai (PVG)
- New York ↔ Reykjavik (KEF)
- New York ↔ Los Angeles (LAX)

"New York" uses the city code `NYC`, which covers JFK, Newark, and LaGuardia. The exact airport for each flight is recorded in the data.

## Phase 1: Modeling

### The question

**Will the cheapest price for a route and departure date be higher 7 days from now?** In traveler terms: book now, or wait a week?

### Choosing the target (`explore.py`)

Exploration showed that day-to-day prices are mostly flat: between consecutive daily observations, the cheapest price was unchanged 79% of the time. A "will it change tomorrow?" model could score 79% accuracy by always answering "no" while learning nothing. Over a 7-day horizon, outcomes are balanced (56% up, 41% down, 3% unchanged), so the target uses 7 days.

### Evaluation setup (`dataset.py`)

- **Time-based split.** Models train on earlier observation days and are tested on the last 5 labeled days, never on a random shuffle, which would let the model learn from prices in the test period.
- **Leakage gap.** A training example's label depends on the price 7 days after it was observed, so training examples stop 7 days before the test period begins.
- **Balanced accuracy** is the main metric: the average of accuracy on "up" cases and on "not up" cases. A model that always guesses one answer scores 50%.
- **Shared dataset.** The baselines and the neural network use identical examples and the same split, so the comparison is fair.

### Baselines (`baseline.py`)

| Model | Balanced accuracy |
|---|---|
| Always predict the majority outcome | 50.0% |
| Days-until rule: "up" if within 40 days of departure (threshold learned from training data) | **59.1%** |
| Most common outcome per route and days-until range | 54.6% |

The days-until rule rediscovered a known airline pricing pattern: fares tend to rise in the last month or so before departure. The route-specific rule did worse because splitting into many small groups made it fit noise in the training data.

### Neural network (`model.py`)

A small feed-forward network in PyTorch (two hidden layers of 32 and 16 units, ReLU, dropout, weight decay), trained with binary cross-entropy. Each configuration is trained with 5 random seeds and reported as a mean and spread, so differences from the baseline can be separated from run-to-run noise.

| Model | Inputs | Balanced accuracy (5 seeds) |
|---|---|---|
| Neural network, all features | 4 numeric + route + day of week (23) | 54.5% ± 0.7 |
| Neural network, numeric features only | days until departure, 3-day price change, price relative to route median (4) | 59.0% ± 0.6 |
| Days-until rule (baseline) | days until departure | 59.1% |

**Ablation.** The full network lost to the baseline by 4.5 points, well outside its seed-to-seed spread. Removing the route and day-of-week inputs closed the gap entirely, which indicates the extra inputs let the network overfit the small training set (about 1,300 examples).

### Conclusion so far

With 26 days of data, a single-threshold rule captures essentially all the predictable signal, and the neural network matches it but does not beat it. The saved model uses the numeric-only configuration. Because collection continues daily, the models will be retrained as data accumulates to see whether route-level patterns become learnable with more history.

Limitations: training currently covers only about 10 observation days, test examples from the same route and date are not fully independent, network settings are reasonable defaults rather than tuned (tuning would require a separate validation set, which becomes practical with more data), and the Taipei routes have too few examples to evaluate on their own.

## Phase 2: Containerization

Each part of the system runs as its own Docker container, so it behaves the same on a laptop as on a cloud server. All three are defined in `docker-compose.yml`.

| Service | Dockerfile | What it does | Type |
|---|---|---|---|
| `collector` | `docker/collector.Dockerfile` | Collects one day of prices | One-off job |
| `train` | `docker/train.Dockerfile` | Trains the model and saves it to `models/` | One-off job |
| `api` | `docker/api.Dockerfile` | Serves predictions over HTTP | Long-running service |

### Inference API (`api.py`)

A FastAPI server that loads the trained model once at startup and answers: *is the cheapest price for this route and departure date likely to be higher in 7 days?*

- `POST /predict`: takes a route, departure date, current price, and optionally the price 3 days ago; returns the probability of a rise and a plain-language recommendation
- `GET /routes`: lists the routes the model was trained on
- `GET /health`: liveness check used by the container health check
- `GET /docs`: interactive test page generated by FastAPI

Example request:

```json
{
  "route": "NYC-LAX",
  "departure_date": "2027-01-15",
  "current_price": 189
}
```

Requests are validated automatically, and the API rejects routes the model never saw and departures within 7 days, which fall outside its training data.

### Design choices

- **Separate images per service.** The collector image contains only `requests` and `python-dotenv`. PyTorch goes only into the training and API images, which use the CPU-only build to avoid several gigabytes of unused GPU libraries.
- **Layer caching.** Each Dockerfile installs dependencies before copying code, so code changes rebuild in seconds without reinstalling packages.
- **No secrets in images.** `.dockerignore` excludes `.env`; the API token is passed in at run time.
- **Volumes with least privilege.** Training mounts the data read-only, and the API mounts the model read-only, so neither can modify what it depends on.
- **No training/serving skew.** The API builds its inputs with the same feature code and saved standardization values used in training. The model file also stores each route's median price and which inputs it was trained with, and the API refuses to start if they don't match the current code.
- **Jobs vs. services.** Compose profiles keep the one-off jobs from starting with `docker compose up`, which starts only the API.
- **Health check.** Docker calls `/health` every 30 seconds and marks the API unhealthy if it stops responding.
- **Reproducibility.** Training inside the container reproduced the local results exactly (59.0% balanced accuracy across the same 5 seeds).

## Data

### Schema (`data/processed/prices_YYYY-MM-DD.csv`)

| Column | Description |
|---|---|
| `observed_date` | Date the price was recorded (UTC). Turns daily snapshots into a time series. |
| `observed_at_utc` | Exact timestamp of the run |
| `origin`, `destination` | City or airport codes as returned by the API |
| `origin_airport`, `destination_airport` | Exact airports for the flight |
| `departure_date` | Flight departure date |
| `days_until_departure` | Departure date minus observed date |
| `price` | Price in USD |
| `airline`, `flight_number` | Carrier details |
| `transfers` | Number of stops |
| `duration_min` | Total trip duration in minutes |

### Source and limitations

- **Source.** Prices come from the Travelpayouts (Aviasales) Data API. The project was originally planned around the Amadeus Self-Service API, which Amadeus shut down in July 2026.
- **Cached prices.** The API returns prices from searches made by Aviasales users in roughly the last 48 hours, not live quotes, so a price can repeat across consecutive days.
- **Code labels.** The API sometimes labels destinations by city code (SHA for Shanghai, REK for Reykjavik) or returns nearby airports (Ontario and Long Beach for Los Angeles, Stewart for New York). `dataset.py` maps these back to the tracked routes.
- **Uneven coverage.** Less-searched routes and dates return fewer prices, so some routes have gaps.
- **One holiday season so far.** Collection began September 27, 2026, so the data captures the 2026 holiday price run-up once. Confirming the pattern repeats requires a second season.
- **Date convention.** Dates are in UTC. The earliest snapshots (Sept 27 to Oct 1) were collected locally using Central time, with a gap from Sept 28 to 30 before automation began.

## Running It Yourself

**Requirements:** Python 3.10+ and a free [Travelpayouts](https://www.travelpayouts.com) account (API token under Profile → API token).

```bash
git clone <this-repo-url>
cd flight-price-tracker
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your token:

```
TRAVELPAYOUTS_TOKEN=your_token_here
```

Collect one snapshot:

```bash
python collector.py
```

Explore the data, run the baselines, and train the neural network:

```bash
python explore.py
python baseline.py
python model.py
```

**With Docker** (requires Docker Desktop), from the project folder:

```bash
docker compose run --rm train      # train the model on data/
docker compose up                  # start the API, then open http://localhost:8000/docs
docker compose down                # stop it
```

`docker compose run --rm collector` runs a test collection into `docker-test-data/`. Add `--build` to any command after changing code.

**To schedule collection with GitHub Actions:** add your token as a repository secret named `TRAVELPAYOUTS_TOKEN` (Settings → Secrets and variables → Actions). The workflow then runs daily and can also be started manually from the Actions tab.

## Planned Architecture

```
[Scheduled Data Collector] --> [S3: raw price data]
                                       |
                                       v
                              [Training Job: PyTorch]
                                       |
                                       v
                              [Trained Model: S3]
                                       |
                                       v
                        [Inference API] <-- [User/request]
```

All components containerized with Docker. Deployed on AWS (S3, EventBridge/Lambda or ECS for scheduling, API Gateway or ECS for serving predictions). Kubernetes orchestration is a stretch goal once the core pipeline is stable.

## Phase Breakdown

### Phase 1: Data Collector + Baseline Model (done)
- Pull flight price data on a schedule for a fixed set of routes
- Store collected data as the dataset grows, committed daily by GitHub Actions
- Explore the data, build baselines, and train a PyTorch model to predict price direction

### Phase 2: Containerization (done)
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

**In use:** Python, Requests, Pandas, PyTorch, FastAPI, Docker, Docker Compose, Travelpayouts Data API, GitHub Actions

**Planned:** AWS (S3, Lambda, EventBridge, API Gateway/ECS, ECR), Kubernetes (stretch)

## Project Structure

```
flight-price-tracker/
├── .github/workflows/collect.yml   # daily scheduled collection
├── collector.py                    # data collector
├── explore.py                      # data exploration and target selection
├── dataset.py                      # labeled examples and time-based split
├── baseline.py                     # simple baseline models
├── model.py                        # PyTorch model
├── api.py                          # FastAPI inference server
├── docker/
│   ├── collector.Dockerfile
│   ├── train.Dockerfile
│   └── api.Dockerfile
├── docker-compose.yml              # runs all three services
├── .dockerignore
├── requirements-collector.txt      # per-image dependencies
├── requirements-train.txt
├── requirements-api.txt
├── data/
│   ├── raw/                        # daily raw API responses (JSON)
│   └── processed/                  # daily cleaned snapshots (CSV)
├── models/                         # trained model files (not committed)
├── .env.example                    # template for the API token
├── requirements.txt
└── README.md
```
