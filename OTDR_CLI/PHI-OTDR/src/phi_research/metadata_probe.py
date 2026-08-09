"""Session-level acquisition and label-confounding probes for development data."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def session_means(bundle: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    """Aggregate windows without allowing long sessions to dominate a probe."""
    features = bundle["features"]
    labels = bundle["labels"].astype(np.int64)
    sessions = bundle["sessions"].astype(str)
    partitions = bundle["partitions"].astype(str)
    date_tokens = bundle["date_tokens"].astype(str)
    source_tokens = bundle["source_tokens"].astype(str)
    eras = bundle["eras"].astype(str)
    unique_sessions = sorted(set(sessions.tolist()))
    rows: dict[str, list[object]] = defaultdict(list)
    for session_id in unique_sessions:
        selected = np.flatnonzero(sessions == session_id)
        if len(set(labels[selected].tolist())) != 1:
            raise ValueError(f"Session spans classes: {session_id}")
        for name, values in (
            ("partition", partitions),
            ("date_token", date_tokens),
            ("source_token", source_tokens),
            ("era", eras),
        ):
            if len(set(values[selected].tolist())) != 1:
                raise ValueError(f"Session spans {name} values: {session_id}")
        rows["features"].append(np.mean(features[selected], axis=0))
        rows["label"].append(int(labels[selected[0]]))
        rows["session"].append(session_id)
        rows["partition"].append(partitions[selected[0]])
        rows["date_token"].append(date_tokens[selected[0]])
        rows["source_token"].append(source_tokens[selected[0]])
        rows["date_source"].append(f"{date_tokens[selected[0]]}|{source_tokens[selected[0]]}")
        rows["era"].append(eras[selected[0]])
    return {
        key: np.stack(value) if key == "features" else np.asarray(value)
        for key, value in rows.items()
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(len(y_true)),
        "classes": int(len(set(y_true.tolist()))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def categorical_to_class_probe(
    table: dict[str, np.ndarray], train: np.ndarray, validation: np.ndarray, fields: tuple[str, ...]
) -> dict[str, object]:
    values = np.column_stack([table[field] for field in fields])
    transformer = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), list(range(len(fields))))]
    )
    model = Pipeline(
        [
            ("one_hot", transformer),
            (
                "model",
                LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=2000, random_state=20260805
                ),
            ),
        ]
    )
    model.fit(values[train], table["label"][train])
    predicted = model.predict(values[validation])
    return {
        "input_fields": list(fields),
        "target": "event_class",
        "validation": _metrics(table["label"][validation], predicted),
        "unseen_validation_values": {
            field: sorted(set(table[field][validation]) - set(table[field][train])) for field in fields
        },
    }


def signal_to_metadata_probe(
    table: dict[str, np.ndarray], train: np.ndarray, validation: np.ndarray, target: str
) -> dict[str, object]:
    known_labels = set(table[target][train].tolist())
    evaluable = validation & np.asarray([value in known_labels for value in table[target]])
    unseen = sorted(set(table[target][validation]) - known_labels)
    if not np.any(evaluable):
        return {"target": target, "evaluable": False, "unseen_validation_labels": unseen}
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=3000, random_state=20260805
                ),
            ),
        ]
    )
    model.fit(table["features"][train], table[target][train])
    predicted = model.predict(table["features"][evaluable])
    return {
        "target": target,
        "evaluable": True,
        "validation": _metrics(table[target][evaluable], predicted),
        "excluded_unseen_validation_sessions": int(np.sum(validation) - np.sum(evaluable)),
        "unseen_validation_labels": unseen,
    }


def inventory_confounding(table: dict[str, np.ndarray]) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in ("date_token", "source_token", "date_source"):
        groups: dict[str, set[int]] = defaultdict(set)
        counts: Counter[str] = Counter()
        for value, label in zip(table[field], table["label"], strict=True):
            groups[str(value)].add(int(label))
            counts[str(value)] += 1
        result[field] = {
            "group_count": len(groups),
            "single_class_group_count": sum(len(classes) == 1 for classes in groups.values()),
            "class_count_distribution": dict(sorted(Counter(len(v) for v in groups.values()).items())),
            "session_count_range": [min(counts.values()), max(counts.values())],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-partition", default="source_train")
    parser.add_argument("--validation-partition", default="source_validation")
    args = parser.parse_args()
    bundle = np.load(args.features, allow_pickle=False)
    try:
        if "target_query" in set(bundle["partitions"].astype(str)):
            raise ValueError("Metadata development probe must not receive target-query features")
        table = session_means(bundle)
    finally:
        bundle.close()
    train = table["partition"] == args.train_partition
    validation = table["partition"] == args.validation_partition
    if not np.any(train) or not np.any(validation):
        raise ValueError("Metadata probe received an empty train or validation partition")
    categorical = [
        categorical_to_class_probe(table, train, validation, ("date_token",)),
        categorical_to_class_probe(table, train, validation, ("source_token",)),
        categorical_to_class_probe(table, train, validation, ("date_token", "source_token")),
    ]
    signal = [
        signal_to_metadata_probe(table, train, validation, target)
        for target in ("date_token", "source_token", "date_source")
    ]
    payload = {
        "protocol": "session-level source-train to source-validation development probe",
        "train_partition": args.train_partition,
        "validation_partition": args.validation_partition,
        "train_sessions": int(np.sum(train)),
        "validation_sessions": int(np.sum(validation)),
        "era": sorted(set(table["era"][train | validation].tolist())),
        "inventory_confounding": inventory_confounding(table),
        "metadata_to_event_class": categorical,
        "signal_to_metadata": signal,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
