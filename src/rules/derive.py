"""Bridge ML insights to credit policy via a small decision tree.

Why a decision tree?
- It produces axis-aligned thresholds that map cleanly to "if-then" rules
  a credit-risk officer can audit and (if needed) override.
- We fit it on the SAME features the LightGBM model uses, but limit depth
  so each leaf corresponds to a short, readable rule.
- Each leaf is annotated with: support (% of population), default rate
  inside the leaf, and lift vs. the base rate. We surface the top-K leaves
  by combined lift × support.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from src.config import SETTINGS
from src.data.loader import load_dataframe
from src.data.preprocess import (
    add_engineered_features,
    build_preprocessor,
    feature_columns,
    get_feature_names_out,
)


@dataclass
class Rule:
    rule_id: int
    conditions: list[str]
    support_pct: float       # share of population reaching this leaf
    default_rate_pct: float  # default rate inside the leaf
    lift: float              # leaf default rate / base default rate
    n_samples: int
    band: str                # "Low" | "Medium" | "High"

    def as_text(self) -> str:
        cond = " AND ".join(self.conditions) if self.conditions else "(no conditions)"
        return (
            f"Rule {self.rule_id}: IF {cond} "
            f"THEN risk = {self.band} "
            f"(support={self.support_pct:.1f}%, "
            f"default_rate={self.default_rate_pct:.1f}%, "
            f"lift={self.lift:.2f}x, n={self.n_samples})"
        )


def _tree_to_rules(tree, feature_names: list[str], base_rate: float) -> list[Rule]:
    """Walk a fitted DecisionTreeClassifier and emit one Rule per leaf."""
    t = tree.tree_
    feature_index = t.feature
    threshold = t.threshold
    children_left = t.children_left
    children_right = t.children_right
    values = t.value  # shape (n_nodes, 1, n_classes)
    n_node_samples = t.n_node_samples
    total = n_node_samples[0]

    rules: list[Rule] = []
    rule_id = 1

    def walk(node: int, conds: list[str]) -> None:
        nonlocal rule_id
        if children_left[node] == _tree.TREE_LEAF:
            counts = values[node][0]
            n = int(n_node_samples[node])
            default_rate = float(counts[1] / max(counts.sum(), 1))
            support = n / total
            lift = default_rate / base_rate if base_rate > 0 else 0.0
            band = (
                "High" if default_rate >= SETTINGS.risk_medium_max
                else "Medium" if default_rate >= SETTINGS.risk_low_max
                else "Low"
            )
            rules.append(Rule(
                rule_id=rule_id,
                conditions=conds.copy(),
                support_pct=round(support * 100, 2),
                default_rate_pct=round(default_rate * 100, 2),
                lift=round(lift, 2),
                n_samples=n,
                band=band,
            ))
            rule_id += 1
            return

        fname = feature_names[feature_index[node]]
        thr = threshold[node]
        # Decoded one-hot columns look like "NAME_INCOME_TYPE_Working".
        # We render them as equality checks against the source column.
        cond_left, cond_right = _format_conditions(fname, thr)
        walk(children_left[node], conds + [cond_left])
        walk(children_right[node], conds + [cond_right])

    walk(0, [])
    return rules


def _format_conditions(feature_name: str, threshold: float) -> tuple[str, str]:
    """Pretty-print a node split. Handles one-hot booleans nicely.

    OneHotEncoder produces names like ``NAME_INCOME_TYPE_Working``. We try
    to match the longest known-categorical prefix, so the source column
    is always recognised even when its value contains underscores or
    spaces.
    """
    from src.data.schema import CATEGORICAL_COLUMNS

    if abs(threshold - 0.5) < 1e-6:
        # Sort by length descending so longer prefixes (e.g. NAME_INCOME_TYPE)
        # win over shorter ones (NAME).
        for src in sorted(CATEGORICAL_COLUMNS, key=len, reverse=True):
            prefix = src + "_"
            if feature_name.startswith(prefix):
                val = feature_name[len(prefix):]
                return (f"{src} != '{val}'", f"{src} == '{val}'")
        # Unknown OHE column; surface the raw name for traceability.
        return (f"NOT {feature_name}", f"{feature_name}")
    return (
        f"{feature_name} <= {round(float(threshold), 3)}",
        f"{feature_name} > {round(float(threshold), 3)}",
    )


def derive_rules(max_depth: int = 4, top_k: int = 12, random_state: int = 42) -> list[Rule]:
    """Fit a small decision tree and emit ranked rules."""
    df = load_dataframe()
    df = add_engineered_features(df)
    y = df["TARGET"].astype(int).values
    X_raw = df[feature_columns()].copy()

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(X_raw)
    feat_names = get_feature_names_out(preprocessor)

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=int(0.02 * len(X)),  # ≥2% support per leaf
        class_weight="balanced",
        random_state=random_state,
    )
    tree.fit(X, y)

    base_rate = float(np.mean(y))
    rules = _tree_to_rules(tree, feat_names, base_rate)

    # Rank by absolute deviation from base rate, weighted by support.
    rules.sort(
        key=lambda r: abs(r.default_rate_pct - base_rate * 100) * r.support_pct,
        reverse=True,
    )
    return rules[:top_k]


def save_rules(rules: list[Rule], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) | {"as_text": r.as_text()} for r in rules]
    path.write_text(json.dumps(payload, indent=2))
    return path


if __name__ == "__main__":
    rules = derive_rules()
    out = save_rules(rules, SETTINGS.models_dir / "rules.json")
    print(f"Saved {len(rules)} rules -> {out}\n")
    for r in rules:
        print(" ", r.as_text())
