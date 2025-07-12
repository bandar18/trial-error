from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    train_days: int = 42,
    val_days: int = 7,
) -> Tuple[LGBMClassifier, RandomForestClassifier, Dict[str, np.ndarray]]:
    """Train base (LightGBM) and meta (RandomForest) models.

    Splits data chronologically: train_days, val_days, remaining == test.
    Returns models and dict containing predictions & y_test.
    """
    start = X.index.min()
    train_end = start + pd.Timedelta(days=train_days)
    val_end = train_end + pd.Timedelta(days=val_days)

    X_train, y_train = X.loc[:train_end], y.loc[:train_end]
    X_val, y_val = X.loc[train_end + pd.Timedelta(minutes=5) : val_end], y.loc[
        train_end + pd.Timedelta(minutes=5) : val_end
    ]
    X_test, y_test = X.loc[val_end + pd.Timedelta(minutes=5) :], y.loc[
        val_end + pd.Timedelta(minutes=5) :
    ]

    base = LGBMClassifier(
        objective="multiclass",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    base.fit(X_train, y_train)

    # Validation predictions for meta-label dataset
    val_proba = base.predict_proba(X_val)
    val_pred = val_proba.argmax(axis=1) - 1  # map to {-1,0,1}
    meta_target = (val_pred == y_val.values).astype(int)

    meta_features = pd.DataFrame(
        {
            "prob_neg": val_proba[:, 0],
            "prob_neu": val_proba[:, 1],
            "prob_pos": val_proba[:, 2],
        },
        index=X_val.index,
    ).join(
        X_val[["r1", "sigma10", "delta_vix", "delta_dxy"]]
    )

    meta = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1,
    )
    meta.fit(meta_features, meta_target)

    # Test set predictions
    test_proba = base.predict_proba(X_test)
    test_pred = test_proba.argmax(axis=1) - 1

    meta_test_features = pd.DataFrame(
        {
            "prob_neg": test_proba[:, 0],
            "prob_neu": test_proba[:, 1],
            "prob_pos": test_proba[:, 2],
        },
        index=X_test.index,
    ).join(
        X_test[["r1", "sigma10", "delta_vix", "delta_dxy"]]
    )
    meta_test_proba = meta.predict_proba(meta_test_features)[:, 1]

    # Logging balanced accuracy on val
    bal_acc = balanced_accuracy_score(y_val, val_pred)
    print(f"[Info] Validation balanced accuracy: {bal_acc:.3f}")

    results = {
        "y_test": y_test.values,
        "base_pred": test_pred,
        "base_proba": test_proba,
        "meta_proba": meta_test_proba,
        "timestamps": X_test.index.values,
    }
    return base, meta, results


def generate_signals(results: Dict[str, np.ndarray], threshold: float = 0.55) -> pd.Series:
    """Translate predictions into trading signals."""
    base_pred = results["base_pred"]
    meta_p = results["meta_proba"]
    ts = pd.to_datetime(results["timestamps"])

    signal = np.zeros_like(base_pred, dtype="int8")
    long_mask = (base_pred == 1) & (meta_p >= threshold)
    short_mask = (base_pred == -1) & (meta_p >= threshold)

    signal[long_mask] = 1
    signal[short_mask] = -1

    return pd.Series(signal, index=ts)