"""Decision-tree rule extraction.

Fits a small decision tree on the same engineered features the LightGBM
model uses, then walks every leaf and turns it into one IF-THEN rule.
The leaf statistics (support, default rate inside the leaf, lift vs.
the base rate) come straight from the tree's own counts.
"""

import json
from dataclasses import asdict, dataclass

import numpy as np
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
    conditions: list
    support_pct: float
    default_rate_pct: float
    lift: float
    n_samples: int
    band: str

    def as_text(self):
        cond = " AND ".join(self.conditions) if self.conditions else "(no conditions)"
        return (
            f"Rule {self.rule_id}: IF {cond} "
            f"THEN risk = {self.band} "
            f"(support={self.support_pct:.1f}%, "
            f"default_rate={self.default_rate_pct:.1f}%, "
            f"lift={self.lift:.2f}x, n={self.n_samples})"
        )


def _format_conditions(feature_name, threshold):
    """Render a tree split nicely. One-hot booleans get equality syntax."""
    from src.data.schema import CATEGORICAL_COLUMNS

    if abs(threshold - 0.5) < 1e-6:
        # OneHotEncoder columns look like NAME_INCOME_TYPE_Working. Match
        # the longest known categorical prefix so values containing
        # underscores / spaces still resolve.
        for src in sorted(CATEGORICAL_COLUMNS, key=len, reverse=True):
            prefix = src + "_"
            if feature_name.startswith(prefix):
                val = feature_name[len(prefix):]
                return (f"{src} != '{val}'", f"{src} == '{val}'")
        # Unrecognised one-hot column - fall back to raw form.
        return (f"NOT {feature_name}", f"{feature_name}")

    return (
        f"{feature_name} <= {round(float(threshold), 3)}",
        f"{feature_name} > {round(float(threshold), 3)}",
    )


def _tree_to_rules(tree, feature_names, base_rate):
    t = tree.tree_
    feature_index = t.feature
    threshold = t.threshold
    children_left = t.children_left
    children_right = t.children_right
    values = t.value
    n_node_samples = t.n_node_samples
    total = n_node_samples[0]

    rules = []
    next_id = [1]

    def walk(node, conds):
        if children_left[node] == _tree.TREE_LEAF:
            counts = values[node][0]
            n = int(n_node_samples[node])
            default_rate = float(counts[1] / max(counts.sum(), 1))
            support = n / total
            lift = default_rate / base_rate if base_rate > 0 else 0.0

            if default_rate >= SETTINGS.risk_medium_max:
                band = "High"
            elif default_rate >= SETTINGS.risk_low_max:
                band = "Medium"
            else:
                band = "Low"

            rules.append(Rule(
                rule_id=next_id[0],
                conditions=conds.copy(),
                support_pct=round(support * 100, 2),
                default_rate_pct=round(default_rate * 100, 2),
                lift=round(lift, 2),
                n_samples=n,
                band=band,
            ))
            next_id[0] += 1
            return

        fname = feature_names[feature_index[node]]
        thr = threshold[node]
        cond_left, cond_right = _format_conditions(fname, thr)
        walk(children_left[node], conds + [cond_left])
        walk(children_right[node], conds + [cond_right])

    walk(0, [])
    return rules


def derive_rules(max_depth=4, top_k=12, random_state=42):
    df = load_dataframe()
    df = add_engineered_features(df)

    y = df["TARGET"].astype(int).values
    X_raw = df[feature_columns()].copy()

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(X_raw)
    feat_names = get_feature_names_out(preprocessor)

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        # At least 2% of the population per leaf - keeps rules meaningful.
        min_samples_leaf=int(0.02 * len(X)),
        class_weight="balanced",
        random_state=random_state,
    )
    tree.fit(X, y)

    base_rate = float(np.mean(y))
    rules = _tree_to_rules(tree, feat_names, base_rate)

    # Rank by how much each leaf shifts the default rate, weighted by support.
    rules.sort(
        key=lambda r: abs(r.default_rate_pct - base_rate * 100) * r.support_pct,
        reverse=True,
    )
    return rules[:top_k]


def save_rules(rules, path):
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
