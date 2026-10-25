"""
model.py - PyTorch model for "will the cheapest price be higher in 7 days?"

Trains a small neural network on the same examples and the same time-based
split as baseline.py, then compares it against the best baseline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from baseline import evaluate, predict_days_rule
from dataset import CURRENT_ROUTES, build_dataset, time_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "price_direction.pt"

NUMERIC_FEATURES = [
    "days_until_departure",
    "recent_change_pct",
    "has_recent_price",
    "price_vs_route_median",
]
SEEDS = [0, 1, 2, 3, 4]  # train several times to measure run-to-run noise
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3  # penalizes large weights, which discourages overfitting

# Experiment switch: include route and day-of-week one-hot inputs or not.
USE_ROUTE_AND_DOW = False


# ── Section 1: Turn dataframe rows into numeric feature matrices ─────────────
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Change over the last 3 days as a percentage, so routes are comparable.
    df["recent_change_pct"] = df.recent_change / df.price
    return df


def fit_scaler(train: pd.DataFrame) -> dict:
    """Compute mean and std from TRAINING data only (no peeking at test)."""
    means = train[NUMERIC_FEATURES].mean()
    stds = train[NUMERIC_FEATURES].std().replace(0, 1.0)
    return {"mean": means.to_dict(), "std": stds.to_dict()}


def to_matrix(df: pd.DataFrame, scaler: dict) -> np.ndarray:
    """Standardized numeric features + one-hot route + one-hot day of week."""
    mean = pd.Series(scaler["mean"])
    std = pd.Series(scaler["std"])
    numeric = ((df[NUMERIC_FEATURES] - mean) / std).to_numpy()

    route_onehot = np.zeros((len(df), len(CURRENT_ROUTES)))
    route_index = df.route.map({r: i for i, r in enumerate(CURRENT_ROUTES)})
    route_onehot[np.arange(len(df)), route_index.to_numpy()] = 1.0

    dow_onehot = np.zeros((len(df), 7))
    dow_onehot[np.arange(len(df)), df.departure_dow.to_numpy()] = 1.0

    if not USE_ROUTE_AND_DOW:
        return numeric.astype(np.float32)
    return np.hstack([numeric, route_onehot, dow_onehot]).astype(np.float32)


# ── Section 2: The network ───────────────────────────────────────────────────
class PriceDirectionNet(nn.Module):
    """A small feed-forward network. Outputs one logit: high means 'up'."""

    def __init__(self, n_inputs: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


# ── Section 3: Training loop ─────────────────────────────────────────────────
def train_model(X: np.ndarray, y: np.ndarray, seed: int) -> PriceDirectionNet:
    torch.manual_seed(seed)
    model = PriceDirectionNet(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)

    model.train()  # turns dropout on
    for _ in range(EPOCHS):
        order = torch.randperm(len(X_t), generator=generator)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[batch]), y_t[batch])
            loss.backward()
            optimizer.step()
    return model


def predict(model: PriceDirectionNet, X: np.ndarray) -> np.ndarray:
    model.eval()  # turns dropout off
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
    return (logits > 0).int().numpy()


# ── Section 4: Run, compare to baseline, save ────────────────────────────────
def main() -> None:
    data = add_derived_features(build_dataset())
    train, test = time_split(data)
    scaler = fit_scaler(train)
    X_train, X_test = to_matrix(train, scaler), to_matrix(test, scaler)

    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows | "
          f"Inputs per example: {X_train.shape[1]}\n")

    baseline = evaluate("Days-until rule (baseline)", test.label, predict_days_rule(train, test))

    runs, models = [], []
    for seed in SEEDS:
        model = train_model(X_train, train.label.to_numpy(), seed)
        preds = pd.Series(predict(model, X_test), index=test.index)
        runs.append(evaluate(f"seed {seed}", test.label, preds))
        models.append(model)

    runs_df = pd.DataFrame(runs)
    print("\nNeural network, one row per training run:")
    print(runs_df.round(3).to_string(index=False))

    mean, std = runs_df.balanced_accuracy.mean(), runs_df.balanced_accuracy.std()
    print(f"\nNeural network balanced accuracy: {mean:.3f} (+/- {std:.3f} across seeds)")
    print(f"Baseline balanced accuracy:       {baseline['balanced_accuracy']:.3f}")
    gap = mean - baseline["balanced_accuracy"]
    print(f"Difference: {gap:+.3f}")

    # Save the first run's model plus everything needed to reuse it later
    # (the inference API in Phase 2 will load this file).
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": models[0].state_dict(),
            "n_inputs": X_train.shape[1],
            "scaler": scaler,
            "routes": CURRENT_ROUTES,
            "numeric_features": NUMERIC_FEATURES,
        },
        MODEL_PATH,
    )
    print(f"\nSaved model to {MODEL_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
