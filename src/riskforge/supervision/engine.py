from typing import Any, Callable, Dict, List, Optional
import numpy as np

ABSTAIN = -1
NON_SIF = 0
SIF_P = 1

class LabelingFunction:
    def __init__(self, name: str, f: Callable[[Any], int]) -> None:
        self.name = name
        self.f = f

    def __call__(self, record: Any) -> int:
        return self.f(record)

def labeling_function(name: Optional[str] = None) -> Callable[[Callable[[Any], int]], LabelingFunction]:
    def decorator(f: Callable[[Any], int]) -> LabelingFunction:
        lf_name = name if name is not None else f.__name__
        return LabelingFunction(name=lf_name, f=f)
    return decorator

class LFApplier:
    def __init__(self, lfs: List[LabelingFunction]) -> None:
        self.lfs = lfs

    def apply(self, records: List[Any]) -> np.ndarray:
        num_records = len(records)
        num_lfs = len(self.lfs)
        L = np.full((num_records, num_lfs), ABSTAIN, dtype=np.int32)
        for i, record in enumerate(records):
            for j, lf in enumerate(self.lfs):
                L[i, j] = lf(record)
        return L

class LFAnalysis:
    def __init__(self, L: np.ndarray, lfs: List[LabelingFunction]) -> None:
        self.L = L
        self.lfs = lfs
        self.num_records, self.num_lfs = L.shape

    def summary(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for j, lf in enumerate(self.lfs):
            col = self.L[:, j]
            non_abstain_mask = col != ABSTAIN
            coverage = float(np.mean(non_abstain_mask)) if self.num_records > 0 else 0.0

            other_cols = np.delete(self.L, j, axis=1)
            other_active = np.any(other_cols != ABSTAIN, axis=1)
            overlaps = float(np.mean(non_abstain_mask & other_active)) if self.num_records > 0 else 0.0

            conflicts = 0
            active_indices = np.where(non_abstain_mask)[0]
            for idx in active_indices:
                val = col[idx]
                row = other_cols[idx]
                row_active = row[row != ABSTAIN]
                if np.any(row_active != val):
                    conflicts += 1
            conflict_rate = float(conflicts / self.num_records) if self.num_records > 0 else 0.0

            stats[lf.name] = {
                "coverage": round(coverage, 4),
                "overlaps": round(overlaps, 4),
                "conflicts": round(conflict_rate, 4),
            }
        return stats

class DawidSkeneLabelModel:
    def __init__(self, num_classes: int = 2, max_iter: int = 100, tol: float = 1e-5) -> None:
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.tol = tol
        self.class_priors = np.zeros(num_classes, dtype=np.float64)
        self.error_rates: Optional[np.ndarray] = None

    def fit(self, L: np.ndarray) -> None:
        N, M = L.shape
        K = self.num_classes

        class_counts = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            class_counts[:, k] = np.sum(np.where(L == k, 1.0, 0.0), axis=1)

        row_sums = class_counts.sum(axis=1, keepdims=True)
        all_abstain = np.where(row_sums.squeeze(-1) == 0)[0]
        row_sums[row_sums == 0] = 1.0
        T = class_counts / row_sums
        T[all_abstain, :] = 1.0 / K

        for _ in range(self.max_iter):
            old_T = T.copy()
            self.class_priors = T.mean(axis=0)

            self.error_rates = np.zeros((M, K, K), dtype=np.float64)
            for j in range(M):
                for k in range(K):
                    weights = T[:, k]
                    denom = weights.sum()
                    if denom == 0:
                        self.error_rates[j, k, :] = 1.0 / K
                        continue
                    for l in range(K):
                        numer = np.sum(weights * (L[:, j] == l))
                        self.error_rates[j, k, l] = numer / denom

            self.error_rates = np.clip(self.error_rates, 1e-6, 1.0 - 1e-6)

            log_T = np.zeros((N, K), dtype=np.float64)
            for k in range(K):
                log_prior = np.log(self.class_priors[k] + 1e-12)
                log_lik = np.zeros(N, dtype=np.float64)
                for j in range(M):
                    col = L[:, j]
                    for l in range(K):
                        mask = col == l
                        log_lik[mask] += np.log(self.error_rates[j, k, l])
                log_T[:, k] = log_prior + log_lik

            max_log = np.max(log_T, axis=1, keepdims=True)
            exp_log = np.exp(log_T - max_log)
            T = exp_log / exp_log.sum(axis=1, keepdims=True)

            if np.max(np.abs(T - old_T)) < self.tol:
                break

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        if self.error_rates is None:
            raise RuntimeError("LabelModel must be fitted before calling predict_proba.")

        N, M = L.shape
        K = self.num_classes
        log_T = np.zeros((N, K), dtype=np.float64)

        for k in range(K):
            log_prior = np.log(self.class_priors[k] + 1e-12)
            log_lik = np.zeros(N, dtype=np.float64)
            for j in range(M):
                col = L[:, j]
                for l in range(K):
                    mask = col == l
                    log_lik[mask] += np.log(self.error_rates[j, k, l])
            log_T[:, k] = log_prior + log_lik

        max_log = np.max(log_T, axis=1, keepdims=True)
        exp_log = np.exp(log_T - max_log)
        return exp_log / exp_log.sum(axis=1, keepdims=True)
