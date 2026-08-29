"""Веса голов HomEns по val F1 и мягкое взвешенное голосование."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


def compute_weights(
    methods_order: Sequence[str],
    val_f1_by_method: Mapping[str, float],
    eps: float = 1e-6,
) -> dict[str, float]:
    """Вес головы обратно пропорционален ошибке ``1 - F1`` на валидации.

    Формула::

        w_m ∝ 1 / (max(0, 1 - F1_m) + eps)

    затем нормализация, чтобы сумма весов была 1.
    """
    if not methods_order:
        raise ValueError("methods_order must be non-empty")
    inv_errs: dict[str, float] = {}
    for method in methods_order:
        err = max(0.0, 1.0 - float(val_f1_by_method[method]))
        inv_errs[method] = 1.0 / (err + eps)
    total = sum(inv_errs.values())
    return {method: inv_errs[method] / total for method in methods_order}


def soft_weighted_proba(
    proba_by_method: Mapping[str, NDArray],
    weights: Mapping[str, float],
    methods_order: Sequence[str],
) -> NDArray[np.float64]:
    """Взвешенная сумма ``predict_proba`` выбранных методов."""
    if not methods_order:
        raise ValueError("methods_order must be non-empty")
    first = np.asarray(proba_by_method[methods_order[0]], dtype=np.float64)
    out = np.zeros_like(first, dtype=np.float64)
    for method in methods_order:
        out += float(weights[method]) * np.asarray(proba_by_method[method], dtype=np.float64)
    return out
