from __future__ import annotations

"""Known-preserving selective classification and OOD score utilities."""

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.special import expit
from scipy.stats import beta, chi2
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite non-empty 2D array.")
    return result


@dataclass(frozen=True)
class PrototypeBank:
    class_ids: tuple[int, ...]
    prototypes: tuple[np.ndarray, ...]
    metric: str

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        *,
        class_ids: Iterable[int] | None = None,
        prototypes_per_class: int = 1,
        metric: str = "cosine",
        seed: int = 42,
    ) -> "PrototypeBank":
        x = _as_2d(embeddings, "embeddings")
        y = np.asarray(labels, dtype=int)
        ids = tuple(sorted(np.unique(y) if class_ids is None else (int(v) for v in class_ids)))
        if len(y) != len(x) or not ids:
            raise ValueError("Labels and embeddings must be aligned and classes non-empty.")
        if not 1 <= prototypes_per_class <= 8:
            raise ValueError("prototypes_per_class must be in 1..8.")
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be cosine or euclidean.")
        rows: list[np.ndarray] = []
        for class_id in ids:
            values = x[y == class_id]
            if len(values) == 0:
                raise ValueError(f"Class {class_id} has no embeddings.")
            count = min(prototypes_per_class, len(values))
            if count == 1:
                centers = values.mean(0, keepdims=True)
            else:
                centers = KMeans(n_clusters=count, random_state=seed + class_id, n_init=10).fit(values).cluster_centers_
            if metric == "cosine":
                centers = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
            rows.append(centers.astype(np.float64))
        return cls(ids, tuple(rows), metric)

    def class_distances(self, embeddings: np.ndarray) -> np.ndarray:
        x = _as_2d(embeddings, "embeddings")
        if self.metric == "cosine":
            x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
        result = []
        for centers in self.prototypes:
            if self.metric == "cosine":
                distance = 1.0 - x @ centers.T
            else:
                distance = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
            result.append(distance.min(1))
        return np.stack(result, axis=1)

    def novelty(self, embeddings: np.ndarray) -> np.ndarray:
        return self.class_distances(embeddings).min(1)


@dataclass(frozen=True)
class DistanceReference:
    features: np.ndarray
    labels: np.ndarray
    mean: np.ndarray
    precision: np.ndarray
    diagonal_precision: np.ndarray

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        *,
        max_reference: int = 2048,
    ) -> "DistanceReference":
        x = _as_2d(embeddings, "embeddings")
        y = np.asarray(labels, dtype=int)
        if len(y) != len(x):
            raise ValueError("Labels and embeddings must be aligned.")
        if max_reference < len(np.unique(y)):
            raise ValueError("max_reference must permit at least one row per class.")
        selected: list[int] = []
        per_class = max(1, max_reference // len(np.unique(y)))
        for class_id in sorted(np.unique(y)):
            candidates = np.flatnonzero(y == class_id)
            selected.extend(candidates[np.linspace(0, len(candidates) - 1, min(per_class, len(candidates)), dtype=int)])
        selected_array = np.asarray(selected[:max_reference], dtype=int)
        covariance = LedoitWolf().fit(x)
        variance = np.var(x, axis=0, ddof=1)
        shrink = 0.1 * np.nanmedian(variance[variance > 0]) if np.any(variance > 0) else 1.0
        diagonal = 1.0 / np.clip(variance + shrink, 1e-8, None)
        return cls(
            x[selected_array].copy(),
            y[selected_array].copy(),
            covariance.location_.copy(),
            covariance.precision_.copy(),
            diagonal,
        )

    def distances(self, query: np.ndarray, *, knn_k: int = 10) -> dict[str, np.ndarray]:
        x = _as_2d(query, "query")
        if knn_k < 1:
            raise ValueError("knn_k must be positive.")
        k = min(knn_k, self.features.shape[0])
        qn = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
        rn = self.features / np.clip(np.linalg.norm(self.features, axis=1, keepdims=True), 1e-12, None)
        try:
            import faiss

            euclidean_index = faiss.IndexFlatL2(self.features.shape[1])
            euclidean_index.add(np.ascontiguousarray(self.features, dtype=np.float32))
            euclidean_distance, _ = euclidean_index.search(np.ascontiguousarray(x, dtype=np.float32), k)
            euclidean_knn = np.sqrt(np.maximum(euclidean_distance, 0)).mean(1)
            cosine_index = faiss.IndexFlatIP(self.features.shape[1])
            cosine_index.add(np.ascontiguousarray(rn, dtype=np.float32))
            cosine_similarity, _ = cosine_index.search(np.ascontiguousarray(qn, dtype=np.float32), k)
            cosine_knn = (1.0 - cosine_similarity).mean(1)
        except ImportError:
            # Bounded-memory exact fallback for environments without FAISS.
            euclidean_rows, cosine_rows = [], []
            reference_norm = np.square(self.features).sum(1)[None, :]
            for start in range(0, len(x), 512):
                stop = start + 512
                query = x[start:stop]
                squared = (
                    np.square(query).sum(1)[:, None]
                    + reference_norm
                    - 2.0 * query @ self.features.T
                )
                euclidean = np.sqrt(np.maximum(squared, 0))
                euclidean_rows.append(np.partition(euclidean, k - 1, axis=1)[:, :k].mean(1))
                cosine = 1.0 - qn[start:stop] @ rn.T
                cosine_rows.append(np.partition(cosine, k - 1, axis=1)[:, :k].mean(1))
            euclidean_knn = np.concatenate(euclidean_rows)
            cosine_knn = np.concatenate(cosine_rows)
        centered = x - self.mean
        mahalanobis = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", centered, self.precision, centered), 0))
        diagonal = np.sqrt(np.maximum((np.square(centered) * self.diagonal_precision).sum(1), 0))
        return {
            "cosine_knn": cosine_knn,
            "euclidean_knn": euclidean_knn,
            "mahalanobis": mahalanobis,
            "diagonal_mahalanobis": diagonal,
        }


def classifier_novelty_components(logits: np.ndarray) -> dict[str, np.ndarray]:
    values = _as_2d(logits, "logits")
    shifted = values - values.max(1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(1, keepdims=True)
    sorted_logits = np.sort(values, axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(1)
    energy = -np.log(np.exp(shifted).sum(1)) - values.max(1)
    return {
        "one_minus_msp": 1.0 - probabilities.max(1),
        "entropy": entropy,
        "negative_margin": -(sorted_logits[:, -1] - sorted_logits[:, -2]),
        "negative_max_logit": -values.max(1),
        "energy": energy,
    }


def assemble_components(
    *,
    logits: np.ndarray,
    embeddings: np.ndarray,
    distance_reference: DistanceReference,
    prototype_bank: PrototypeBank,
    physics_residual: np.ndarray | None = None,
    knn_k: int = 10,
) -> tuple[tuple[str, ...], np.ndarray]:
    values = classifier_novelty_components(logits)
    values.update(distance_reference.distances(embeddings, knn_k=knn_k))
    values["multi_prototype"] = prototype_bank.novelty(embeddings)
    if physics_residual is not None:
        residual = np.asarray(physics_residual, dtype=float)
        if residual.shape != (len(logits),) or not np.isfinite(residual).all():
            raise ValueError("physics_residual must be finite and aligned.")
        values["physics_residual"] = residual
    names = tuple(values)
    return names, np.column_stack([values[name] for name in names])


@dataclass(frozen=True)
class EmpiricalCDFNormalizer:
    names: tuple[str, ...]
    reference: tuple[np.ndarray, ...]

    @classmethod
    def fit(cls, components: np.ndarray, names: Iterable[str]) -> "EmpiricalCDFNormalizer":
        values = _as_2d(components, "components")
        names = tuple(names)
        if len(names) != values.shape[1] or len(set(names)) != len(names):
            raise ValueError("Component names must be unique and aligned.")
        return cls(names, tuple(np.sort(values[:, index]) for index in range(values.shape[1])))

    def transform(self, components: np.ndarray) -> np.ndarray:
        values = _as_2d(components, "components")
        if values.shape[1] != len(self.reference):
            raise ValueError("Component count differs from fitted normalizer.")
        result = np.empty_like(values)
        for index, reference in enumerate(self.reference):
            result[:, index] = (np.searchsorted(reference, values[:, index], side="right") + 0.5) / (
                len(reference) + 1.0
            )
        return np.clip(result, 1e-6, 1 - 1e-6)


def fuse_scores(
    normalized: np.ndarray,
    *,
    method: Literal["confidence", "best_single", "weighted", "sirc", "meta_p", "robust_regret"],
    weights: Iterable[float] | None = None,
    confidence_index: int = 0,
    ood_index: int | None = None,
) -> np.ndarray:
    values = _as_2d(normalized, "normalized")
    if not 0 <= confidence_index < values.shape[1]:
        raise ValueError("confidence_index is invalid.")
    if method == "confidence":
        return values[:, confidence_index]
    if method == "best_single":
        index = values.shape[1] - 1 if ood_index is None else ood_index
        return values[:, index]
    if method == "weighted":
        w = np.ones(values.shape[1]) if weights is None else np.asarray(tuple(weights), dtype=float)
        if w.shape != (values.shape[1],) or np.allclose(w, 0):
            raise ValueError("Fusion weights must align and be nonzero.")
        w = w / np.abs(w).sum()
        return values @ w
    if method == "sirc":
        index = values.shape[1] - 1 if ood_index is None else ood_index
        competence_novelty = values[:, confidence_index]
        familiarity_novelty = values[:, index]
        # Add familiarity evidence mainly where competence is not already decisive.
        return competence_novelty + familiarity_novelty * (1.0 - competence_novelty)
    if method == "meta_p":
        p_values = np.clip(1.0 - values, 1e-12, 1)
        statistic = -2.0 * np.log(p_values).sum(1)
        return chi2.cdf(statistic, 2 * values.shape[1])
    if method == "robust_regret":
        return 0.5 * np.median(values, axis=1) + 0.5 * np.max(values, axis=1)
    raise ValueError(f"Unknown fusion method: {method}")


class KnownPseudoUnseenSelector:
    """Small 2D selector fitted only on known/pseudo-unseen inner tasks."""

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1.0, class_weight="balanced", random_state=0)
        self.fitted = False

    def fit(self, competence_novelty: np.ndarray, familiarity_novelty: np.ndarray, pseudo_unseen: np.ndarray) -> "KnownPseudoUnseenSelector":
        x = np.column_stack((competence_novelty, familiarity_novelty))
        y = np.asarray(pseudo_unseen, dtype=int)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Selector fitting requires known and pseudo-unseen examples.")
        self.model.fit(x, y)
        self.fitted = True
        return self

    def predict_score(self, competence_novelty: np.ndarray, familiarity_novelty: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Selector is not fitted.")
        x = np.column_stack((competence_novelty, familiarity_novelty))
        return self.model.predict_proba(x)[:, 1]


@dataclass(frozen=True)
class JointThreshold:
    threshold: float
    normal_far_cap: float
    known_acceptance_floor: float
    calibration_normal_far: float
    calibration_known_fault_acceptance: float
    normal_groups: int
    known_fault_groups: int
    mode: str


def _higher_quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q, method="higher"))


def fit_joint_threshold(
    score: np.ndarray,
    labels: np.ndarray,
    *,
    normal_far_cap: float = 0.0125,
    known_acceptance_floor: float = 0.95,
    mode: Literal["empirical", "conformal"] = "empirical",
) -> JointThreshold:
    values = np.asarray(score, dtype=float)
    y = np.asarray(labels, dtype=int)
    if values.shape != y.shape or not np.isfinite(values).all():
        raise ValueError("Finite aligned score and labels are required.")
    normal = values[y == 0]
    faults = values[y != 0]
    if len(normal) < 2 or len(faults) < 2:
        raise ValueError("Joint calibration requires normal and known-fault groups.")
    if not 0 < normal_far_cap < 1 or not 0 < known_acceptance_floor < 1:
        raise ValueError("Constraints must be probabilities in (0,1).")
    normal_q = 1.0 - normal_far_cap
    if mode == "conformal":
        # Split-conformal upper quantile with finite-sample rank correction.
        normal_q = min(1.0, np.ceil((len(normal) + 1) * normal_q) / len(normal))
    elif mode != "empirical":
        raise ValueError("mode must be empirical or conformal.")
    threshold = max(
        _higher_quantile(normal, normal_q),
        _higher_quantile(faults, known_acceptance_floor),
    )
    return JointThreshold(
        threshold=threshold,
        normal_far_cap=normal_far_cap,
        known_acceptance_floor=known_acceptance_floor,
        calibration_normal_far=float((normal > threshold).mean()),
        calibration_known_fault_acceptance=float((faults <= threshold).mean()),
        normal_groups=len(normal),
        known_fault_groups=len(faults),
        mode=mode,
    )


def _group_equal_weights(group_ids: Iterable[str]) -> np.ndarray:
    groups = np.asarray(tuple(str(value) for value in group_ids))
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def _weighted_higher_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    target = np.clip(q, 0, 1) * cumulative[-1]
    return float(sorted_values[min(np.searchsorted(cumulative, target, side="left"), len(values) - 1)])


def fit_joint_threshold_grouped(
    score: np.ndarray,
    labels: np.ndarray,
    group_ids: Iterable[str],
    *,
    normal_far_cap: float = 0.0125,
    known_acceptance_floor: float = 0.95,
    mode: Literal["empirical", "conformal"] = "empirical",
) -> JointThreshold:
    """Fit a threshold with each exact input group receiving equal total weight."""
    values = np.asarray(score, dtype=float)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(tuple(str(value) for value in group_ids))
    if values.shape != y.shape or groups.shape != y.shape or not np.isfinite(values).all():
        raise ValueError("Grouped calibration requires aligned finite values, labels, and groups.")
    for group in np.unique(groups):
        if len(np.unique(y[groups == group])) != 1:
            raise ValueError("A calibration group cannot contain conflicting labels.")
    weights = _group_equal_weights(groups)
    normal = y == 0
    faults = y != 0
    if normal.sum() < 2 or faults.sum() < 2:
        raise ValueError("Grouped joint calibration requires normal and known-fault examples.")
    normal_q = 1.0 - normal_far_cap
    normal_groups = int(np.unique(groups[normal]).size)
    if mode == "conformal":
        normal_q = min(
            1.0,
            np.ceil((normal_groups + 1) * normal_q) / normal_groups,
        )
    elif mode != "empirical":
        raise ValueError("mode must be empirical or conformal.")
    threshold = max(
        _weighted_higher_quantile(values[normal], weights[normal], normal_q),
        _weighted_higher_quantile(
            values[faults], weights[faults], known_acceptance_floor
        ),
    )

    def weighted_rate(mask: np.ndarray, event: np.ndarray) -> float:
        selected_weights = weights[mask]
        return float(np.average(event[mask], weights=selected_weights))

    rejected = values > threshold
    return JointThreshold(
        threshold=threshold,
        normal_far_cap=normal_far_cap,
        known_acceptance_floor=known_acceptance_floor,
        calibration_normal_far=weighted_rate(normal, rejected),
        calibration_known_fault_acceptance=weighted_rate(faults, ~rejected),
        normal_groups=normal_groups,
        known_fault_groups=int(np.unique(groups[faults]).size),
        mode=f"group_equal_weight_{mode}",
    )


def evaluate_grouped_operating_point(
    score: np.ndarray,
    labels: np.ndarray,
    predicted: np.ndarray,
    group_ids: Iterable[str],
    *,
    holdout: tuple[int, int],
    calibration: JointThreshold,
    bootstrap_iterations: int = 2000,
    seed: int = 20260725,
) -> dict[str, object]:
    """Evaluate rates with equal group weight and a group-bootstrap FAR interval."""
    values = np.asarray(score, dtype=float)
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(predicted, dtype=int)
    groups = np.asarray(tuple(str(value) for value in group_ids))
    if not (values.shape == y.shape == pred.shape == groups.shape):
        raise ValueError("Grouped evaluation arrays must align.")
    weights = _group_equal_weights(groups)
    rejected = values > calibration.threshold
    normal = y == 0
    unknown = np.isin(y, holdout)
    known_fault = (~normal) & (~unknown)
    known = ~unknown

    def rate(mask: np.ndarray, event: np.ndarray) -> float:
        return float(np.average(event[mask], weights=weights[mask]))

    group_frame = {}
    for group in np.unique(groups[normal]):
        mask = groups == group
        group_frame[group] = float(rejected[mask].mean())
    normal_group_rates = np.asarray(list(group_frame.values()), dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = np.mean(
        normal_group_rates[
            rng.integers(
                0,
                len(normal_group_rates),
                size=(bootstrap_iterations, len(normal_group_rates)),
            )
        ],
        axis=1,
    )
    per_fault = {
        str(class_id): rate(y == class_id, rejected)
        for class_id in holdout
    }
    accepted_known = known & (~rejected)
    return {
        "group_weighted_normal_far": rate(normal, rejected),
        "group_weighted_normal_far_bootstrap_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "normal_independent_groups": len(normal_group_rates),
        "normal_groups_with_any_rejection_rate": float(
            np.mean(normal_group_rates > 0)
        ),
        "group_weighted_known_fault_acceptance": rate(known_fault, ~rejected),
        "group_weighted_unknown_recall": rate(unknown, rejected),
        "group_weighted_worst_fault_recall": min(per_fault.values()),
        "group_weighted_per_fault_recall": per_fault,
        "group_weighted_accepted_known_accuracy": rate(
            accepted_known, pred == y
        ) if accepted_known.any() else float("nan"),
        "group_constraints_met": bool(
            rate(normal, rejected) <= calibration.normal_far_cap
            and rate(known_fault, ~rejected) >= calibration.known_acceptance_floor
        ),
        "bootstrap_iterations": bootstrap_iterations,
    }


def binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Invalid binomial counts.")
    alpha = 1 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def evaluate_joint_operating_point(
    score: np.ndarray,
    labels: np.ndarray,
    predicted: np.ndarray,
    *,
    holdout: tuple[int, int],
    calibration: JointThreshold,
) -> dict[str, object]:
    values = np.asarray(score, dtype=float)
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(predicted, dtype=int)
    rejected = values > calibration.threshold
    normal = y == 0
    unknown = np.isin(y, holdout)
    known_fault = (~unknown) & (~normal)
    known = ~unknown
    far_count = int(rejected[normal].sum())
    far_interval = binomial_interval(far_count, int(normal.sum()))
    accepted_known = known & (~rejected)
    if any(not np.any(y == class_id) for class_id in holdout):
        raise ValueError("Every held-out class must occur in the evaluation labels.")
    per_fault = {str(class_id): float(rejected[y == class_id].mean()) for class_id in holdout}
    normal_far = float(rejected[normal].mean())
    known_acceptance = float((~rejected[known_fault]).mean())
    return {
        "threshold": calibration.threshold,
        "normal_far": normal_far,
        "normal_far_count": far_count,
        "normal_count": int(normal.sum()),
        "normal_far_clopper_pearson_95": list(far_interval),
        "known_fault_acceptance": known_acceptance,
        "unknown_recall": float(rejected[unknown].mean()),
        "worst_fault_recall": min(per_fault.values()),
        "per_fault_recall": per_fault,
        "accepted_known_accuracy": float((pred[accepted_known] == y[accepted_known]).mean()) if accepted_known.any() else float("nan"),
        "overall_known_accuracy_rejection_failure": float(((pred == y) & (~rejected) & known).sum() / max(known.sum(), 1)),
        "constraints_met": bool(
            normal_far <= calibration.normal_far_cap and known_acceptance >= calibration.known_acceptance_floor
        ),
    }
