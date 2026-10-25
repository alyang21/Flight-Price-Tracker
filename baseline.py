"""
baseline.py - Simple baselines for "will the price be higher in 7 days?"

Any later model (including the PyTorch one) has to beat these numbers on the
same test set, or it isn't earning its complexity.
"""

import pandas as pd

from dataset import HORIZON_DAYS, build_dataset, time_split

DAY_BUCKETS = [0, 14, 30, 60, 90, 180, 400]


# ── Section 1: Metrics ───────────────────────────────────────────────────────
def evaluate(name: str, y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Accuracy, plus balanced accuracy (average of per-class accuracy)."""
    y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
    accuracy = (y_true == y_pred).mean()
    up = y_true == 1
    recall_up = (y_pred[up] == 1).mean() if up.any() else float("nan")
    recall_not_up = (y_pred[~up] == 0).mean() if (~up).any() else float("nan")
    return {
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": (recall_up + recall_not_up) / 2,
    }


# ── Section 2: Baselines ─────────────────────────────────────────────────────
def predict_majority(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Always predict whichever outcome was most common in training."""
    majority = int(train.label.mean() >= 0.5)
    return pd.Series(majority, index=test.index)


def predict_days_rule(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Predict 'up' when the flight is within a threshold of departure.
    The threshold is chosen as whichever works best on the training set."""
    best_threshold, best_acc = None, -1.0
    for threshold in range(10, 200, 5):
        acc = ((train.days_until_departure <= threshold).astype(int) == train.label).mean()
        if acc > best_acc:
            best_threshold, best_acc = threshold, acc
    print(f"  days rule: predict 'up' within {best_threshold} days of departure")
    return (test.days_until_departure <= best_threshold).astype(int)


def predict_group_rate(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Predict the outcome most common in training for the same route and
    days-until-departure range. Falls back to the overall majority."""
    def bucket(d):
        return pd.cut(d.days_until_departure, DAY_BUCKETS)

    rates = train.groupby([train.route, bucket(train)], observed=True).label.mean()
    keys = pd.MultiIndex.from_arrays([test.route, bucket(test)])
    test_rates = rates.reindex(keys).to_numpy()
    fallback = train.label.mean()
    test_rates = pd.Series(test_rates, index=test.index).fillna(fallback)
    return (test_rates >= 0.5).astype(int)


# ── Section 3: Run ───────────────────────────────────────────────────────────
def main() -> None:
    data = build_dataset()
    train, test = time_split(data)

    print(f"Labeled examples: {len(data):,} (horizon {HORIZON_DAYS} days)")
    print(f"Train: {len(train):,} rows, observed {train.observed_date.min().date()} "
          f"to {train.observed_date.max().date()}")
    print(f"Test:  {len(test):,} rows, observed {test.observed_date.min().date()} "
          f"to {test.observed_date.max().date()}")
    print(f"Share 'up' - train: {train.label.mean():.0%}, test: {test.label.mean():.0%}\n")

    results = [
        evaluate("Always majority", test.label, predict_majority(train, test)),
        evaluate("Days-until rule", test.label, predict_days_rule(train, test)),
        evaluate("Route + days rate", test.label, predict_group_rate(train, test)),
    ]
    print()
    print(pd.DataFrame(results).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
