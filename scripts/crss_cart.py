"""Leakage-safe, survey-weighted CART evaluation utilities.

The functions in this module are deliberately independent of the project
notebooks.  They provide a reusable development-time evaluation path with:

* fold-local preprocessing;
* hyperparameter selection inside each outer training fold;
* survey-weighted Brier loss for selection and evaluation;
* sigmoid calibration learned from cross-fitted predictions generated only
  inside the corresponding outer training fold; and
* optional group-preserving folds for PSU sensitivity analysis.

These utilities evaluate development data.  They do not fit or inspect a
reserved final-test partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


DEFAULT_PARAM_GRID: Mapping[str, Sequence[Any]] = {
    "max_depth": (4, 6),
    "min_samples_leaf": (40, 120),
    "ccp_alpha": (0.0, 0.0005),
}

_ALLOWED_SEARCH_PARAMETERS = {
    "ccp_alpha",
    "criterion",
    "max_depth",
    "max_features",
    "max_leaf_nodes",
    "min_impurity_decrease",
    "min_samples_leaf",
    "min_samples_split",
    "min_weight_fraction_leaf",
    "splitter",
}


@dataclass(frozen=True)
class HyperparameterSelection:
    """Result of an inner-CV, survey-weighted CART search."""

    best_params: dict[str, Any]
    best_weighted_brier: float
    search_results: pd.DataFrame
    best_oof_probability: pd.Series


@dataclass(frozen=True)
class NestedCartEvaluation:
    """Artifacts from a complete nested CART out-of-fold evaluation.

    Attributes
    ----------
    predictions:
        One row per input observation with its outer fold, observed outcome,
        survey weight, raw probability, and calibrated probability.
    fold_summary:
        Per-outer-fold model-selection, performance, calibration, and tree-size
        diagnostics.
    search_results:
        Every hyperparameter candidate and its pooled inner out-of-fold Brier
        score, identified by outer fold.
    leaf_diagnostics:
        Training and validation support, event rate, probability, and Brier
        score for every terminal leaf used in every outer fold.
    raw_weighted_brier / calibrated_weighted_brier:
        Pooled outer out-of-fold survey-weighted scores.
    grouped_outer:
        Whether groups were kept intact in outer, inner, and calibration folds.
    """

    predictions: pd.DataFrame
    fold_summary: pd.DataFrame
    search_results: pd.DataFrame
    leaf_diagnostics: pd.DataFrame
    raw_weighted_brier: float
    calibrated_weighted_brier: float
    grouped_outer: bool

    def metric_table(self) -> pd.DataFrame:
        """Return the two pooled Brier scores in tidy form."""

        return pd.DataFrame(
            {
                "prediction": ["raw", "sigmoid_calibrated"],
                "weighted_brier": [
                    self.raw_weighted_brier,
                    self.calibrated_weighted_brier,
                ],
            }
        )


@dataclass(frozen=True)
class PsuSensitivityEvaluation:
    """Row-wise and PSU-grouped nested evaluations plus their comparison."""

    row_wise: NestedCartEvaluation
    psu_grouped: NestedCartEvaluation
    comparison: pd.DataFrame


class SigmoidProbabilityCalibrator:
    """Weighted sigmoid calibration for a one-dimensional raw probability.

    A logistic model learns ``sigmoid(intercept + coefficient * raw_p)``.
    In nested evaluation, ``fit`` receives only cross-fitted predictions from
    the outer training partition.  The large ``C`` supplies negligible but
    numerically stabilizing L2 regularization.
    """

    def __init__(self, *, max_iter: int = 1_000, c: float = 1_000_000.0) -> None:
        self.max_iter = max_iter
        self.c = c
        self.model_: LogisticRegression | None = None

    def fit(
        self,
        raw_probability: Sequence[float] | np.ndarray,
        outcome: Sequence[int] | np.ndarray,
        sample_weight: Sequence[float] | np.ndarray,
    ) -> "SigmoidProbabilityCalibrator":
        """Fit the weighted sigmoid mapping and return ``self``."""

        probability = _validate_probability(raw_probability)
        target = np.asarray(outcome)
        weights = _validate_weight_array(sample_weight, len(target))
        if len(probability) != len(target):
            raise ValueError("raw_probability and outcome must have equal length")
        if set(np.unique(target)) != {0, 1}:
            raise ValueError("Sigmoid calibration requires both binary outcome classes")

        model = LogisticRegression(
            C=self.c,
            solver="lbfgs",
            max_iter=self.max_iter,
        )
        model.fit(
            probability.reshape(-1, 1),
            target.astype("int8", copy=False),
            sample_weight=_normalized_fit_weights(weights),
        )
        self.model_ = model
        return self

    def predict(self, raw_probability: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return calibrated positive-class probabilities."""

        if self.model_ is None:
            raise RuntimeError("The sigmoid calibrator must be fitted before prediction")
        probability = _validate_probability(raw_probability)
        return self.model_.predict_proba(probability.reshape(-1, 1))[:, 1]

    @property
    def intercept_(self) -> float:
        """Fitted sigmoid intercept."""

        if self.model_ is None:
            raise RuntimeError("The sigmoid calibrator is not fitted")
        return float(self.model_.intercept_[0])

    @property
    def coefficient_(self) -> float:
        """Fitted coefficient multiplying the raw probability."""

        if self.model_ is None:
            raise RuntimeError("The sigmoid calibrator is not fitted")
        return float(self.model_.coef_[0, 0])


def make_cart_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Create an unfitted CART preprocessor.

    Numeric medians are learned from the current training fold.  Categorical
    missing values are retained as categories by ``OneHotEncoder`` and unseen
    validation categories are ignored.  Sparse one-hot output keeps the
    40,000-row development workflow memory-efficient.
    """

    numeric, categorical = _validate_feature_names(
        numeric_features,
        categorical_features,
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                list(numeric),
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                list(categorical),
            )
        )

    return ColumnTransformer(
        transformers,
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def make_cart_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    tree_params: Mapping[str, Any] | None = None,
    random_state: int = 678,
) -> Pipeline:
    """Create a fresh, unfitted preprocessing and CART pipeline."""

    params = _validate_tree_params(tree_params or {})
    return Pipeline(
        [
            ("prepare", make_cart_preprocessor(numeric_features, categorical_features)),
            ("classifier", _make_tree(params, random_state=random_state)),
        ]
    )


def select_cart_hyperparameters(
    features: pd.DataFrame,
    outcome: pd.Series | Sequence[int] | np.ndarray,
    sample_weight: pd.Series | Sequence[float] | np.ndarray,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    inner_splits: int = 3,
    random_state: int = 678,
    groups: pd.Series | Sequence[Any] | np.ndarray | None = None,
    grouped: bool = False,
) -> HyperparameterSelection:
    """Select CART parameters using pooled weighted inner-OOF Brier loss.

    Preprocessing is fitted separately in every inner training fold.  Within
    one fold its transformed matrices are safely reused across tree-only
    hyperparameter candidates, substantially reducing runtime.
    """

    prepared = _prepare_inputs(
        features,
        outcome,
        sample_weight,
        numeric_features,
        categorical_features,
        groups,
        require_groups=grouped,
    )
    return _select_cart_hyperparameters_prepared(
        prepared.features,
        prepared.outcome,
        prepared.sample_weight,
        numeric_features=prepared.numeric_features,
        categorical_features=prepared.categorical_features,
        param_grid=param_grid,
        inner_splits=inner_splits,
        random_state=random_state,
        groups=prepared.groups,
        grouped=grouped,
    )


def evaluate_nested_cart(
    features: pd.DataFrame,
    outcome: pd.Series | Sequence[int] | np.ndarray,
    sample_weight: pd.Series | Sequence[float] | np.ndarray,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    calibration_splits: int = 3,
    random_state: int = 678,
    groups: pd.Series | Sequence[Any] | np.ndarray | None = None,
    grouped_outer: bool = False,
) -> NestedCartEvaluation:
    """Run leakage-safe nested CART selection and sigmoid calibration.

    For each outer fold this function performs the following operations using
    only the outer training partition:

    1. select tree hyperparameters with inner cross-validation;
    2. generate a fresh set of cross-fitted probabilities for calibration;
    3. fit one weighted sigmoid calibrator to those probabilities; and
    4. fit a fresh selected pipeline to all outer training rows.

    The final pipeline then produces raw and calibrated probabilities for the
    untouched outer validation fold.  When ``grouped_outer=True``, ``groups``
    are kept intact in outer, inner-search, and calibration folds.
    """

    _validate_split_count("outer_splits", outer_splits)
    _validate_split_count("inner_splits", inner_splits)
    _validate_split_count("calibration_splits", calibration_splits)
    prepared = _prepare_inputs(
        features,
        outcome,
        sample_weight,
        numeric_features,
        categorical_features,
        groups,
        require_groups=grouped_outer,
    )

    x = prepared.features
    y = prepared.outcome
    weights = prepared.sample_weight
    group_values = prepared.groups
    candidates = _parameter_candidates(param_grid)

    outer_cv = _make_splitter(
        n_splits=outer_splits,
        grouped=grouped_outer,
        random_state=random_state,
    )
    outer_indices = _split_indices(outer_cv, x, y, group_values, grouped_outer)

    raw_oof = np.full(len(x), np.nan, dtype="float64")
    calibrated_oof = np.full(len(x), np.nan, dtype="float64")
    outer_fold = np.full(len(x), -1, dtype="int16")
    fold_rows: list[dict[str, Any]] = []
    search_frames: list[pd.DataFrame] = []
    leaf_frames: list[pd.DataFrame] = []

    for fold_number, (train_positions, validation_positions) in enumerate(
        outer_indices,
        start=1,
    ):
        x_train = x.iloc[train_positions]
        x_validation = x.iloc[validation_positions]
        y_train = y.iloc[train_positions]
        y_validation = y.iloc[validation_positions]
        weight_train = weights.iloc[train_positions]
        weight_validation = weights.iloc[validation_positions]
        group_train = (
            None if group_values is None else group_values.iloc[train_positions]
        )

        _require_both_classes(y_train, f"outer training fold {fold_number}")
        selection_seed = random_state + 1_000 * fold_number
        selection = _select_cart_hyperparameters_prepared(
            x_train,
            y_train,
            weight_train,
            numeric_features=prepared.numeric_features,
            categorical_features=prepared.categorical_features,
            param_grid=candidates,
            inner_splits=inner_splits,
            random_state=selection_seed,
            groups=group_train,
            grouped=grouped_outer,
        )
        fold_search = selection.search_results.copy()
        fold_search.insert(0, "outer_fold", fold_number)
        search_frames.append(fold_search)

        calibration_probability = _cross_fitted_cart_probability(
            x_train,
            y_train,
            weight_train,
            numeric_features=prepared.numeric_features,
            categorical_features=prepared.categorical_features,
            tree_params=selection.best_params,
            n_splits=calibration_splits,
            random_state=random_state + 100_000 + fold_number,
            groups=group_train,
            grouped=grouped_outer,
        )
        calibrator = SigmoidProbabilityCalibrator().fit(
            calibration_probability,
            y_train.to_numpy(),
            weight_train.to_numpy(),
        )

        model = make_cart_pipeline(
            prepared.numeric_features,
            prepared.categorical_features,
            tree_params=selection.best_params,
            random_state=random_state + fold_number,
        )
        model.fit(
            x_train,
            y_train,
            classifier__sample_weight=_normalized_fit_weights(
                weight_train.to_numpy()
            ),
        )
        fold_raw_probability = _pipeline_positive_probability(model, x_validation)
        fold_calibrated_probability = calibrator.predict(fold_raw_probability)

        raw_oof[validation_positions] = fold_raw_probability
        calibrated_oof[validation_positions] = fold_calibrated_probability
        outer_fold[validation_positions] = fold_number

        classifier = model.named_steps["classifier"]
        fold_rows.append(
            {
                "outer_fold": fold_number,
                "n_training": len(train_positions),
                "n_validation": len(validation_positions),
                "training_weight": float(weight_train.sum()),
                "validation_weight": float(weight_validation.sum()),
                "best_params": selection.best_params.copy(),
                "inner_weighted_brier": selection.best_weighted_brier,
                "calibration_oof_weighted_brier": _weighted_brier(
                    y_train,
                    calibration_probability,
                    weight_train,
                ),
                "raw_weighted_brier": _weighted_brier(
                    y_validation,
                    fold_raw_probability,
                    weight_validation,
                ),
                "calibrated_weighted_brier": _weighted_brier(
                    y_validation,
                    fold_calibrated_probability,
                    weight_validation,
                ),
                "sigmoid_intercept": calibrator.intercept_,
                "sigmoid_coefficient": calibrator.coefficient_,
                "tree_depth": int(classifier.get_depth()),
                "terminal_leaves": int(classifier.get_n_leaves()),
                "raw_probability_one_count": int(
                    np.count_nonzero(fold_raw_probability == 1.0)
                ),
                "raw_probability_zero_count": int(
                    np.count_nonzero(fold_raw_probability == 0.0)
                ),
            }
        )
        leaf_frames.append(
            _leaf_diagnostics(
                model,
                calibrator,
                x_train,
                y_train,
                weight_train,
                x_validation,
                y_validation,
                weight_validation,
                outer_fold=fold_number,
            )
        )

    if np.isnan(raw_oof).any() or np.isnan(calibrated_oof).any():
        raise RuntimeError("Every observation must receive both outer-OOF probabilities")
    if np.any(outer_fold < 1):
        raise RuntimeError("Every observation must receive exactly one outer fold")

    prediction_data: dict[str, Any] = {
        "outer_fold": outer_fold,
        "outcome": y.to_numpy(),
        "sample_weight": weights.to_numpy(),
        "raw_probability": raw_oof,
        "calibrated_probability": calibrated_oof,
    }
    if group_values is not None:
        prediction_data["group"] = group_values.to_numpy()
    predictions = pd.DataFrame(prediction_data, index=x.index)

    return NestedCartEvaluation(
        predictions=predictions,
        fold_summary=pd.DataFrame(fold_rows),
        search_results=pd.concat(search_frames, ignore_index=True),
        leaf_diagnostics=pd.concat(leaf_frames, ignore_index=True),
        raw_weighted_brier=_weighted_brier(y, raw_oof, weights),
        calibrated_weighted_brier=_weighted_brier(
            y,
            calibrated_oof,
            weights,
        ),
        grouped_outer=grouped_outer,
    )


def evaluate_cart_with_psu_sensitivity(
    features: pd.DataFrame,
    outcome: pd.Series | Sequence[int] | np.ndarray,
    sample_weight: pd.Series | Sequence[float] | np.ndarray,
    psu_groups: pd.Series | Sequence[Any] | np.ndarray,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    calibration_splits: int = 3,
    random_state: int = 678,
) -> PsuSensitivityEvaluation:
    """Compare row-wise evaluation with a PSU-preserving sensitivity run.

    The grouped run keeps each supplied physical PSU intact at every level of
    model selection and calibration.  For CRSS design-based confidence
    intervals, callers must separately use ``PSUSTRAT`` and ``PSU_VAR``;
    group cross-validation is a generalization sensitivity, not a replacement
    for survey variance estimation.
    """

    common = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "param_grid": param_grid,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "calibration_splits": calibration_splits,
        "random_state": random_state,
        "groups": psu_groups,
    }
    row_wise = evaluate_nested_cart(
        features,
        outcome,
        sample_weight,
        grouped_outer=False,
        **common,
    )
    psu_grouped = evaluate_nested_cart(
        features,
        outcome,
        sample_weight,
        grouped_outer=True,
        **common,
    )
    comparison = pd.DataFrame(
        {
            "outer_strategy": ["row_wise", "psu_grouped"],
            "raw_weighted_brier": [
                row_wise.raw_weighted_brier,
                psu_grouped.raw_weighted_brier,
            ],
            "calibrated_weighted_brier": [
                row_wise.calibrated_weighted_brier,
                psu_grouped.calibrated_weighted_brier,
            ],
        }
    )
    comparison["calibration_brier_change"] = (
        comparison["raw_weighted_brier"]
        - comparison["calibrated_weighted_brier"]
    )
    return PsuSensitivityEvaluation(row_wise, psu_grouped, comparison)


@dataclass(frozen=True)
class _PreparedInputs:
    features: pd.DataFrame
    outcome: pd.Series
    sample_weight: pd.Series
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    groups: pd.Series | None


def _prepare_inputs(
    features: pd.DataFrame,
    outcome: pd.Series | Sequence[int] | np.ndarray,
    sample_weight: pd.Series | Sequence[float] | np.ndarray,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    groups: pd.Series | Sequence[Any] | np.ndarray | None,
    *,
    require_groups: bool,
) -> _PreparedInputs:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if not features.index.is_unique:
        raise ValueError("features must have a unique row index")
    if not features.columns.is_unique:
        raise ValueError("features must have unique column names")
    numeric, categorical = _validate_feature_names(
        numeric_features,
        categorical_features,
    )
    requested = numeric + categorical
    missing = [column for column in requested if column not in features.columns]
    if missing:
        raise ValueError(f"Feature columns are missing: {missing}")

    x = features.loc[:, list(requested)].copy()
    y = _align_series(outcome, x.index, "outcome")
    if y.isna().any():
        raise ValueError("outcome contains missing values")
    if set(pd.unique(y)) != {0, 1}:
        raise ValueError("outcome must contain both binary values 0 and 1")
    y = y.astype("int8")

    weights = _align_series(sample_weight, x.index, "sample_weight").astype(
        "float64"
    )
    _validate_weight_array(weights.to_numpy(), len(x))

    group_series: pd.Series | None = None
    if groups is not None:
        group_series = _align_series(groups, x.index, "groups")
        if group_series.isna().any():
            raise ValueError("groups contains missing values")
    if require_groups and group_series is None:
        raise ValueError("groups is required when grouped evaluation is requested")

    return _PreparedInputs(
        x,
        y,
        weights,
        numeric,
        categorical,
        group_series,
    )


def _select_cart_hyperparameters_prepared(
    features: pd.DataFrame,
    outcome: pd.Series,
    sample_weight: pd.Series,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    param_grid: Mapping[str, Sequence[Any]] | list[dict[str, Any]] | None,
    inner_splits: int,
    random_state: int,
    groups: pd.Series | None,
    grouped: bool,
) -> HyperparameterSelection:
    _validate_split_count("inner_splits", inner_splits)
    candidates = _parameter_candidates(param_grid)
    splitter = _make_splitter(
        n_splits=inner_splits,
        grouped=grouped,
        random_state=random_state,
    )
    splits = _split_indices(splitter, features, outcome, groups, grouped)
    candidate_probability = np.full(
        (len(candidates), len(features)),
        np.nan,
        dtype="float64",
    )

    for inner_fold, (train_positions, validation_positions) in enumerate(
        splits,
        start=1,
    ):
        y_train = outcome.iloc[train_positions]
        _require_both_classes(y_train, f"inner training fold {inner_fold}")
        weight_train = sample_weight.iloc[train_positions]
        preprocessor = make_cart_preprocessor(numeric_features, categorical_features)
        transformed_train = preprocessor.fit_transform(features.iloc[train_positions])
        transformed_validation = preprocessor.transform(
            features.iloc[validation_positions]
        )

        for candidate_number, params in enumerate(candidates):
            model = _make_tree(
                params,
                random_state=random_state + inner_fold,
            )
            model.fit(
                transformed_train,
                y_train,
                sample_weight=_normalized_fit_weights(weight_train.to_numpy()),
            )
            candidate_probability[candidate_number, validation_positions] = (
                _tree_positive_probability(model, transformed_validation)
            )

    if np.isnan(candidate_probability).any():
        raise RuntimeError("Inner CV did not predict every row for every candidate")

    scores = np.array(
        [
            _weighted_brier(outcome, probability, sample_weight)
            for probability in candidate_probability
        ],
        dtype="float64",
    )
    stable_order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype="int64")
    ranks[stable_order] = np.arange(1, len(scores) + 1)
    best_number = int(stable_order[0])

    rows: list[dict[str, Any]] = []
    for candidate_number, (params, score, rank) in enumerate(
        zip(candidates, scores, ranks, strict=True),
        start=1,
    ):
        row: dict[str, Any] = {
            "candidate": candidate_number,
            "weighted_brier": float(score),
            "rank": int(rank),
            "params": params.copy(),
        }
        row.update({f"param_{name}": value for name, value in params.items()})
        rows.append(row)

    return HyperparameterSelection(
        best_params=candidates[best_number].copy(),
        best_weighted_brier=float(scores[best_number]),
        search_results=pd.DataFrame(rows).sort_values(
            ["rank", "candidate"],
            ignore_index=True,
        ),
        best_oof_probability=pd.Series(
            candidate_probability[best_number],
            index=features.index,
            name="inner_oof_probability",
        ),
    )


def _cross_fitted_cart_probability(
    features: pd.DataFrame,
    outcome: pd.Series,
    sample_weight: pd.Series,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    tree_params: Mapping[str, Any],
    n_splits: int,
    random_state: int,
    groups: pd.Series | None,
    grouped: bool,
) -> np.ndarray:
    splitter = _make_splitter(
        n_splits=n_splits,
        grouped=grouped,
        random_state=random_state,
    )
    splits = _split_indices(splitter, features, outcome, groups, grouped)
    probability = np.full(len(features), np.nan, dtype="float64")

    for fold_number, (train_positions, validation_positions) in enumerate(
        splits,
        start=1,
    ):
        y_train = outcome.iloc[train_positions]
        _require_both_classes(y_train, f"calibration training fold {fold_number}")
        pipeline = make_cart_pipeline(
            numeric_features,
            categorical_features,
            tree_params=tree_params,
            random_state=random_state + fold_number,
        )
        pipeline.fit(
            features.iloc[train_positions],
            y_train,
            classifier__sample_weight=_normalized_fit_weights(
                sample_weight.iloc[train_positions].to_numpy()
            ),
        )
        probability[validation_positions] = _pipeline_positive_probability(
            pipeline,
            features.iloc[validation_positions],
        )

    if np.isnan(probability).any():
        raise RuntimeError("Calibration CV did not predict every outer-training row")
    return probability


def _leaf_diagnostics(
    model: Pipeline,
    calibrator: SigmoidProbabilityCalibrator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    weight_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    weight_validation: pd.Series,
    *,
    outer_fold: int,
) -> pd.DataFrame:
    prepare = model.named_steps["prepare"]
    classifier = model.named_steps["classifier"]
    train_leaf = classifier.apply(prepare.transform(x_train))
    validation_transformed = prepare.transform(x_validation)
    validation_leaf = classifier.apply(validation_transformed)
    validation_raw = _tree_positive_probability(classifier, validation_transformed)
    validation_calibrated = calibrator.predict(validation_raw)

    train_frame = pd.DataFrame(
        {
            "leaf_id": train_leaf,
            "outcome": y_train.to_numpy(),
            "weight": weight_train.to_numpy(),
        }
    )
    validation_frame = pd.DataFrame(
        {
            "leaf_id": validation_leaf,
            "outcome": y_validation.to_numpy(),
            "weight": weight_validation.to_numpy(),
            "raw_probability": validation_raw,
            "calibrated_probability": validation_calibrated,
        }
    )

    rows: list[dict[str, Any]] = []
    for leaf_id, validation_group in validation_frame.groupby(
        "leaf_id",
        sort=True,
    ):
        training_group = train_frame.loc[train_frame["leaf_id"].eq(leaf_id)]
        raw_probability = float(validation_group["raw_probability"].iloc[0])
        calibrated_probability = float(
            validation_group["calibrated_probability"].iloc[0]
        )
        rows.append(
            {
                "outer_fold": outer_fold,
                "leaf_id": int(leaf_id),
                "n_training": len(training_group),
                "training_weight": float(training_group["weight"].sum()),
                "training_event_rate": _weighted_average(
                    training_group["outcome"],
                    training_group["weight"],
                ),
                "n_validation": len(validation_group),
                "validation_weight": float(validation_group["weight"].sum()),
                "validation_event_rate": _weighted_average(
                    validation_group["outcome"],
                    validation_group["weight"],
                ),
                "raw_probability": raw_probability,
                "calibrated_probability": calibrated_probability,
                "raw_weighted_brier": _weighted_brier(
                    validation_group["outcome"],
                    validation_group["raw_probability"],
                    validation_group["weight"],
                ),
                "calibrated_weighted_brier": _weighted_brier(
                    validation_group["outcome"],
                    validation_group["calibrated_probability"],
                    validation_group["weight"],
                ),
            }
        )
    return pd.DataFrame(rows)


def _parameter_candidates(
    param_grid: Mapping[str, Sequence[Any]] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if isinstance(param_grid, list):
        if not param_grid:
            raise ValueError("param_grid must produce at least one candidate")
        candidates = [_validate_tree_params(candidate) for candidate in param_grid]
        return [candidate.copy() for candidate in candidates]

    grid = DEFAULT_PARAM_GRID if param_grid is None else param_grid
    if not isinstance(grid, Mapping):
        raise TypeError("param_grid must be a mapping or a candidate list")
    unknown = set(grid) - _ALLOWED_SEARCH_PARAMETERS
    if unknown:
        raise ValueError(f"Unsupported CART search parameter(s): {sorted(unknown)}")
    try:
        candidates = [dict(candidate) for candidate in ParameterGrid(grid)]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid param_grid: {error}") from error
    if not candidates:
        raise ValueError("param_grid must produce at least one candidate")
    return [_validate_tree_params(candidate) for candidate in candidates]


def _validate_tree_params(params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(params) - _ALLOWED_SEARCH_PARAMETERS
    if unknown:
        raise ValueError(f"Unsupported CART parameter(s): {sorted(unknown)}")
    return dict(params)


def _make_tree(
    params: Mapping[str, Any],
    *,
    random_state: int,
) -> DecisionTreeClassifier:
    settings: dict[str, Any] = {"criterion": "gini", "random_state": random_state}
    settings.update(_validate_tree_params(params))
    return DecisionTreeClassifier(**settings)


def _make_splitter(*, n_splits: int, grouped: bool, random_state: int) -> Any:
    _validate_split_count("n_splits", n_splits)
    if grouped:
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def _split_indices(
    splitter: Any,
    features: pd.DataFrame,
    outcome: pd.Series,
    groups: pd.Series | None,
    grouped: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if grouped:
        if groups is None:
            raise ValueError("groups is required for grouped cross-validation")
        return list(splitter.split(features, outcome, groups))
    return list(splitter.split(features, outcome))


def _pipeline_positive_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    probability = model.predict_proba(features)
    return probability[:, _positive_class_position(classifier.classes_)]


def _tree_positive_probability(model: DecisionTreeClassifier, features: Any) -> np.ndarray:
    probability = model.predict_proba(features)
    return probability[:, _positive_class_position(model.classes_)]


def _positive_class_position(classes: np.ndarray) -> int:
    positions = np.flatnonzero(np.asarray(classes) == 1)
    if len(positions) != 1:
        raise ValueError("A fitted CART model must contain positive class 1")
    return int(positions[0])


def _validate_feature_names(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric = tuple(numeric_features)
    categorical = tuple(categorical_features)
    if not numeric and not categorical:
        raise ValueError("At least one predictor is required")
    all_features = numeric + categorical
    if any(not isinstance(name, str) or not name for name in all_features):
        raise ValueError("Every feature name must be a non-empty string")
    if len(set(numeric)) != len(numeric):
        raise ValueError("numeric_features contains duplicates")
    if len(set(categorical)) != len(categorical):
        raise ValueError("categorical_features contains duplicates")
    overlap = set(numeric) & set(categorical)
    if overlap:
        raise ValueError(f"Numeric and categorical features overlap: {sorted(overlap)}")
    return numeric, categorical


def _align_series(values: Any, index: pd.Index, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        if not values.index.is_unique:
            raise ValueError(f"{name} must have a unique index")
        if values.index.equals(index):
            result = values.copy()
        else:
            missing_index = index.difference(values.index)
            if len(missing_index):
                raise ValueError(f"{name} is missing rows present in features")
            result = values.reindex(index)
    else:
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != len(index):
            raise ValueError(f"{name} must be one-dimensional with {len(index)} rows")
        result = pd.Series(array, index=index, name=name)
    result.name = name
    return result


def _validate_weight_array(values: Any, expected_length: int) -> np.ndarray:
    weights = np.asarray(values, dtype="float64")
    if weights.ndim != 1 or len(weights) != expected_length:
        raise ValueError("sample_weight must be one-dimensional and match outcome")
    if not np.isfinite(weights).all():
        raise ValueError("sample_weight must contain only finite values")
    if np.any(weights <= 0):
        raise ValueError("sample_weight values must all be positive")
    return weights


def _validate_probability(values: Any) -> np.ndarray:
    probability = np.asarray(values, dtype="float64")
    if probability.ndim != 1:
        raise ValueError("probability input must be one-dimensional")
    if not np.isfinite(probability).all():
        raise ValueError("probability input must contain only finite values")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("probability values must lie within [0, 1]")
    return probability


def _normalized_fit_weights(weights: np.ndarray) -> np.ndarray:
    values = _validate_weight_array(weights, len(weights))
    return values / values.mean()


def _weighted_brier(outcome: Any, probability: Any, sample_weight: Any) -> float:
    target = np.asarray(outcome)
    prediction = _validate_probability(probability)
    weights = _validate_weight_array(sample_weight, len(target))
    if len(prediction) != len(target):
        raise ValueError("outcome and probability must have equal length")
    return float(brier_score_loss(target, prediction, sample_weight=weights))


def _weighted_average(values: Any, sample_weight: Any) -> float:
    data = np.asarray(values, dtype="float64")
    weights = _validate_weight_array(sample_weight, len(data))
    return float(np.average(data, weights=weights))


def _validate_split_count(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 2:
        raise ValueError(f"{name} must be an integer of at least 2")


def _require_both_classes(outcome: pd.Series, context: str) -> None:
    if set(pd.unique(outcome)) != {0, 1}:
        raise ValueError(f"{context} does not contain both outcome classes")


__all__ = [
    "DEFAULT_PARAM_GRID",
    "HyperparameterSelection",
    "NestedCartEvaluation",
    "PsuSensitivityEvaluation",
    "SigmoidProbabilityCalibrator",
    "evaluate_cart_with_psu_sensitivity",
    "evaluate_nested_cart",
    "make_cart_pipeline",
    "make_cart_preprocessor",
    "select_cart_hyperparameters",
]
