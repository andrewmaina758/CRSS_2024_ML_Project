"""Design-based confidence intervals for CRSS weighted estimates.

The functions in this module use a with-replacement Taylor linearization
variance estimator. PSUSTRAT identifies variance strata and PSU_VAR identifies
primary sampling units within strata. No finite-population correction or
automatic single-PSU adjustment is applied.

The implementation is intentionally strict. Missing values, non-positive
weights, and strata containing fewer than two observed variance PSUs raise an
error instead of silently changing the analytical population or understating
uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

__all__ = [
    "SurveyEstimate",
    "paired_difference_ci",
    "weighted_mean_ci",
]


@dataclass(frozen=True)
class SurveyEstimate:
    """A survey-weighted point estimate and Taylor-linearized interval."""

    estimate: float
    variance: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    degrees_of_freedom: int
    n_observations: int
    n_strata: int
    n_psus: int
    estimand: str
    method: str = "with-replacement Taylor linearization"


def weighted_mean_ci(
    values: Iterable[float],
    weights: Iterable[float],
    strata: Iterable[Any],
    psu: Iterable[Any],
    *,
    confidence_level: float = 0.95,
) -> SurveyEstimate:
    """Estimate a survey-weighted mean and design-based confidence interval.

    Parameters
    ----------
    values:
        Numeric observation-level values.
    weights:
        Positive CRSS sampling weights, normally WEIGHT.
    strata:
        Variance-stratum identifiers, normally PSUSTRAT.
    psu:
        Variance-PSU identifiers, normally PSU_VAR. Identifiers are interpreted
        within strata, so the same label may be reused in different strata.
    confidence_level:
        Two-sided confidence level strictly between zero and one.

    Returns
    -------
    SurveyEstimate
        The ratio-estimator mean, its with-replacement Taylor variance, and a
        Student-t confidence interval using design degrees of freedom equal to
        the number of stratum-PSU pairs minus the number of strata.

    Notes
    -----
    For mean r = sum(w*y) / sum(w), the observation-level linearized variable
    is w * (y - r) / sum(w). These values are summed within each stratum-PSU,
    and the within-stratum dispersion of PSU totals estimates the variance.
    The interval is not clipped to the natural range of values.
    """

    _validate_confidence_level(confidence_level)
    _validate_series_alignment(
        values=values,
        weights=weights,
        strata=strata,
        psu=psu,
    )

    value_array = _as_numeric_vector(values, "values")
    weight_array = _as_numeric_vector(weights, "weights")
    stratum_array = _as_identifier_vector(strata, "strata")
    psu_array = _as_identifier_vector(psu, "psu")

    _validate_equal_lengths(
        values=value_array,
        weights=weight_array,
        strata=stratum_array,
        psu=psu_array,
    )
    if value_array.size == 0:
        raise ValueError("Survey inputs must contain at least one observation")
    if not np.isfinite(value_array).all():
        raise ValueError("values must contain only finite, non-missing numbers")
    if not np.isfinite(weight_array).all():
        raise ValueError("weights must contain only finite, non-missing numbers")
    if np.any(weight_array <= 0):
        raise ValueError("weights must be strictly positive")

    weight_sum = float(weight_array.sum(dtype=np.float64))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("The total survey weight must be finite and positive")

    estimate = float(
        np.dot(weight_array.astype(np.float64), value_array.astype(np.float64))
        / weight_sum
    )
    if not np.isfinite(estimate):
        raise ValueError("The weighted mean is not finite")

    frame = pd.DataFrame(
        {
            "stratum": stratum_array,
            "psu": psu_array,
            "linearized": weight_array * (value_array - estimate) / weight_sum,
        }
    )
    return _linearized_interval(
        frame,
        estimate=estimate,
        confidence_level=confidence_level,
        estimand="survey-weighted mean",
    )


def paired_difference_ci(
    first: Iterable[float],
    second: Iterable[float],
    weights: Iterable[float],
    strata: Iterable[Any],
    psu: Iterable[Any],
    *,
    confidence_level: float = 0.95,
) -> SurveyEstimate:
    """Estimate the weighted mean paired difference first - second.

    This is appropriate for comparing two models on the same crashes. For
    losses such as Brier loss, a positive estimate means the first model has
    greater (worse) average loss; a negative estimate favors the first model.
    The pairing must be preserved row by row.
    """

    _validate_series_alignment(
        first=first,
        second=second,
        weights=weights,
        strata=strata,
        psu=psu,
    )
    first_array = _as_numeric_vector(first, "first")
    second_array = _as_numeric_vector(second, "second")
    _validate_equal_lengths(first=first_array, second=second_array)

    difference = first_array - second_array
    result = weighted_mean_ci(
        difference,
        weights,
        strata,
        psu,
        confidence_level=confidence_level,
    )
    return SurveyEstimate(
        estimate=result.estimate,
        variance=result.variance,
        standard_error=result.standard_error,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        confidence_level=result.confidence_level,
        degrees_of_freedom=result.degrees_of_freedom,
        n_observations=result.n_observations,
        n_strata=result.n_strata,
        n_psus=result.n_psus,
        estimand="survey-weighted paired mean difference (first - second)",
    )


def _linearized_interval(
    frame: pd.DataFrame,
    *,
    estimate: float,
    confidence_level: float,
    estimand: str,
) -> SurveyEstimate:
    psu_totals = (
        frame.groupby(["stratum", "psu"], sort=False, observed=True)[
            "linearized"
        ]
        .sum()
        .rename("total")
        .reset_index()
    )
    psus_per_stratum = psu_totals.groupby(
        "stratum", sort=False, observed=True
    ).size()
    singleton_strata = psus_per_stratum[psus_per_stratum < 2]
    if not singleton_strata.empty:
        labels = ", ".join(repr(value) for value in singleton_strata.index)
        raise ValueError(
            "Taylor linearization requires at least two observed PSUs in "
            f"every stratum; singleton stratum/strata: {labels}"
        )

    centered = psu_totals["total"] - psu_totals.groupby(
        "stratum", sort=False, observed=True
    )["total"].transform("mean")
    multiplier = psu_totals["stratum"].map(
        psus_per_stratum / (psus_per_stratum - 1)
    )
    variance = float(np.sum(multiplier.to_numpy() * centered.to_numpy() ** 2))
    if variance < 0 or not np.isfinite(variance):
        raise ValueError("The Taylor-linearized variance is invalid")

    n_strata = int(psus_per_stratum.size)
    n_psus = int(psu_totals.shape[0])
    degrees_of_freedom = n_psus - n_strata
    if degrees_of_freedom <= 0:
        raise ValueError("Survey design degrees of freedom must be positive")

    standard_error = float(np.sqrt(variance))
    critical_value = float(
        student_t.ppf(0.5 + confidence_level / 2.0, degrees_of_freedom)
    )
    if not np.isfinite(critical_value):
        raise ValueError("The confidence-interval critical value is invalid")
    margin = critical_value * standard_error

    return SurveyEstimate(
        estimate=estimate,
        variance=variance,
        standard_error=standard_error,
        ci_lower=estimate - margin,
        ci_upper=estimate + margin,
        confidence_level=float(confidence_level),
        degrees_of_freedom=degrees_of_freedom,
        n_observations=int(frame.shape[0]),
        n_strata=n_strata,
        n_psus=n_psus,
        estimand=estimand,
    )


def _validate_confidence_level(confidence_level: float) -> None:
    if isinstance(confidence_level, (bool, np.bool_)):
        raise TypeError("confidence_level must be a real number")
    try:
        level = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise TypeError("confidence_level must be a real number") from exc
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1")


def _validate_series_alignment(**vectors: Iterable[Any]) -> None:
    series = [
        (name, vector)
        for name, vector in vectors.items()
        if isinstance(vector, pd.Series)
    ]
    if len(series) < 2:
        return
    reference_name, reference = series[0]
    for name, vector in series[1:]:
        if not reference.index.equals(vector.index):
            raise ValueError(
                "Pandas Series inputs must have identical indexes in the same "
                f"order; {name!r} does not align with {reference_name!r}"
            )


def _as_numeric_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    try:
        return array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc


def _as_identifier_vector(values: Iterable[Any], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if pd.isna(array).any():
        raise ValueError(f"{name} must not contain missing identifiers")
    try:
        for value in array:
            hash(value)
    except TypeError as exc:
        raise TypeError(f"{name} identifiers must be hashable") from exc
    return array


def _validate_equal_lengths(**vectors: np.ndarray) -> None:
    lengths = {name: len(vector) for name, vector in vectors.items()}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Survey inputs must have equal lengths; received {detail}")
