"""
api.py - Inference API for the Flight Price Tracker.

Loads the trained model once at startup and answers:
"Is the cheapest price for this route and departure date likely to be higher
in 7 days?"

Run locally:  uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for an interactive test page.
"""

from datetime import date, datetime, timezone

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dataset import HORIZON_DAYS
from model import MODEL_PATH, USE_ROUTE_AND_DOW, PriceDirectionNet, to_matrix

# ── Section 1: Load the model once, when the server starts ───────────────────
if not MODEL_PATH.exists():
    raise SystemExit(f"No trained model at {MODEL_PATH.name}. Run model.py first.")

checkpoint = torch.load(MODEL_PATH, weights_only=False)
if checkpoint["use_route_and_dow"] != USE_ROUTE_AND_DOW:
    # The saved model and the current feature code must agree, or the inputs
    # would be built differently from how the model was trained.
    raise SystemExit("Model file and model.py disagree on USE_ROUTE_AND_DOW. Retrain.")

model = PriceDirectionNet(checkpoint["n_inputs"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()

SCALER = checkpoint["scaler"]
ROUTES = checkpoint["routes"]
ROUTE_MEDIANS = checkpoint["route_medians"]

app = FastAPI(title="Flight Price Direction API")


# ── Section 2: Request and response formats ──────────────────────────────────
class PredictionRequest(BaseModel):
    route: str = Field(examples=["NYC-TPE"])
    departure_date: date = Field(examples=["2027-01-15"])
    current_price: float = Field(gt=0, examples=[850.0])
    price_3_days_ago: float | None = Field(default=None, gt=0, examples=[820.0])


class PredictionResponse(BaseModel):
    route: str
    departure_date: date
    days_until_departure: int
    probability_price_rises: float
    prediction: str


# ── Section 3: Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    """Simple check that the server is up. Cloud load balancers call this."""
    return {"status": "ok"}


@app.get("/routes")
def routes() -> list[str]:
    return ROUTES


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    route = req.route.upper()
    if route not in ROUTES:
        raise HTTPException(400, f"Unknown route {route}. See /routes.")

    today = datetime.now(timezone.utc).date()
    days_until = (req.departure_date - today).days
    if days_until <= HORIZON_DAYS:
        # The model was only trained on flights more than 7 days out.
        raise HTTPException(400, f"Departure must be more than {HORIZON_DAYS} days away.")

    recent_change = (
        req.current_price - req.price_3_days_ago if req.price_3_days_ago else 0.0
    )
    # Build a one-row dataframe with the same columns training used, then reuse
    # the exact same feature code (to_matrix) so training and serving match.
    row = pd.DataFrame([{
        "route": route,
        "price": req.current_price,
        "days_until_departure": days_until,
        "departure_dow": req.departure_date.weekday(),
        "recent_change_pct": recent_change / req.current_price,
        "has_recent_price": int(req.price_3_days_ago is not None),
        "price_vs_route_median": req.current_price / ROUTE_MEDIANS[route],
    }])
    X = torch.from_numpy(to_matrix(row, SCALER))

    with torch.no_grad():
        probability = torch.sigmoid(model(X)).item()

    return PredictionResponse(
        route=route,
        departure_date=req.departure_date,
        days_until_departure=days_until,
        probability_price_rises=round(probability, 3),
        prediction=(
            "likely to rise within 7 days: consider booking now"
            if probability >= 0.5
            else "not likely to rise within 7 days: waiting is reasonable"
        ),
    )
