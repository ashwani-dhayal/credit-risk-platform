"""Train the LightGBM credit-risk model."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import SETTINGS
from src.data.loader import load_dataframe
from src.data.preprocess import (
    add_engineered_features,
    build_preprocessor,
    feature_columns,
    get_feature_names_out,
)


@dataclass
class TrainingResult:
    roc_auc: float
    pr_auc: float
    ks_statistic: float
    f1_at_optimal_threshold: float
    optimal_threshold: float
    confusion_matrix: list
    n_train: int
    n_val: int
    default_rate: float
    model_path: str
    feature_names: list
    feature_importances: dict
    training_seconds: float


def _ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def train(random_state=42):
    """End-to-end training run. Persists the pipeline + metrics."""
    t0 = time.time()
    df = load_dataframe()
    df = add_engineered_features(df)

    y = df["TARGET"].astype(int).values
    X = df[feature_columns()].copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    preprocessor = build_preprocessor()
    # scale_pos_weight handles the ~8% class imbalance. We compute it
    # from the training fold only (no val leak).
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    lgbm = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", lgbm),
    ])
    pipeline.fit(X_train, y_train)

    proba_val = pipeline.predict_proba(X_val)[:, 1]
    auc = float(roc_auc_score(y_val, proba_val))
    pr_auc = float(average_precision_score(y_val, proba_val))
    ks = _ks_statistic(y_val, proba_val)

    # Pick the threshold that maximises Youden's J on the validation set.
    fpr, tpr, thresholds = roc_curve(y_val, proba_val)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    best_threshold = float(thresholds[best_idx])
    preds = (proba_val >= best_threshold).astype(int)
    f1 = float(f1_score(y_val, preds))
    cm = confusion_matrix(y_val, preds).tolist()

    feat_names = get_feature_names_out(pipeline.named_steps["preprocessor"])
    importances_arr = pipeline.named_steps["classifier"].feature_importances_
    importances = dict(
        sorted(
            zip(feat_names, importances_arr.astype(float).tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    SETTINGS.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": feature_columns(),
            "feature_names_out": feat_names,
            "optimal_threshold": best_threshold,
        },
        SETTINGS.model_path,
    )

    result = TrainingResult(
        roc_auc=round(auc, 4),
        pr_auc=round(pr_auc, 4),
        ks_statistic=round(ks, 4),
        f1_at_optimal_threshold=round(f1, 4),
        optimal_threshold=round(best_threshold, 4),
        confusion_matrix=cm,
        n_train=int(len(X_train)),
        n_val=int(len(X_val)),
        default_rate=round(float(y.mean()), 4),
        model_path=str(SETTINGS.model_path),
        feature_names=feat_names,
        feature_importances={k: round(v, 2) for k, v in list(importances.items())[:25]},
        training_seconds=round(time.time() - t0, 2),
    )

    metrics_path = SETTINGS.models_dir / "metrics.json"
    metrics_path.write_text(json.dumps(asdict(result), indent=2))

    report_path = SETTINGS.models_dir / "classification_report.txt"
    report_path.write_text(
        classification_report(y_val, preds, target_names=["repaid", "defaulted"])
    )
    return result


if __name__ == "__main__":
    res = train()
    print(json.dumps(asdict(res), indent=2)[:2000])
