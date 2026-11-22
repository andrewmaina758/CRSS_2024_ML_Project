"""Run all development candidate models without reading the reserved test partition."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


SEED = 678
THRESHOLD = 0.50
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned_crashes.csv"
RESULTS = ROOT / "results"

CONTEXT = [
    "MONTH", "DAY_WEEK", "HOUR", "LGT_COND", "WEATHER",
    "RELJCT2", "TYP_INT", "WRK_ZONE", "VE_FORMS", "PVH_INVL",
]
VEHICLE = [
    "max_posted_mph", "speed_limit_unknown", "speed_limit_special",
    "wet_or_water", "snow_ice_or_slush", "curve", "motorcycle",
    "large_truck", "oldest_model_year", "model_year_unknown",
]
PERSON = [
    "pedestrian", "pedalcyclist", "youngest_driver_age",
    "oldest_driver_age", "driver_age_status", "unknown_occupant_role",
]
FEATURE_SETS = {
    "context": CONTEXT,
    "combined": CONTEXT + VEHICLE + PERSON,
}
NUMERIC = {
    "context": ["VE_FORMS", "PVH_INVL"],
    "combined": [
        "VE_FORMS", "PVH_INVL", "max_posted_mph",
        "oldest_model_year", "youngest_driver_age", "oldest_driver_age",
    ],
}


def weighted_brier(outcome, probability, weight):
    return brier_score_loss(outcome, probability, sample_weight=weight)


def make_pipeline(method, features, numeric_features, settings):
    numeric = [column for column in features if column in numeric_features]
    categorical = [column for column in features if column not in numeric]
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    ]

    if method == "Logistic regression":
        numeric_steps.append(("scale", StandardScaler()))
        classifier = LogisticRegression(
            solver="newton-cholesky",
            max_iter=80,
            tol=1e-7,
            **settings,
        )
    elif method == "CART":
        classifier = DecisionTreeClassifier(
            criterion="gini",
            random_state=SEED,
            **settings,
        )
    elif method == "Random forest":
        classifier = RandomForestClassifier(
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
            **settings,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    dtype = np.float64 if method == "Logistic regression" else np.float32
    prepare = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=dtype),
            categorical,
        ),
    ])
    return Pipeline([
        ("prepare", prepare),
        ("classifier", classifier),
    ])


def main():
    crashes = pd.read_csv(DATA_PATH).set_index("CASENUM")
    y = crashes["injury_involved"]

    development_ids, _ = train_test_split(
        crashes.index,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )
    development_ids = np.sort(development_ids)
    development = crashes.loc[development_ids].copy()

    folds = pd.Series(-1, index=development.index, dtype="int8")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_number, (_, validation_positions) in enumerate(
        splitter.split(development.index, development["injury_involved"]),
        start=1,
    ):
        folds.loc[development.index[validation_positions]] = fold_number

    cart_settings = [
        {"max_depth": depth, "min_samples_leaf": leaf, "ccp_alpha": pruning}
        for depth in [5, 10]
        for leaf in [40, 120]
        for pruning in [0.0, 0.0002]
    ]
    logistic_settings = [{"C": c} for c in [0.1, 1.0, 10.0]]
    forest_settings = [
        {"n_estimators": 80, "max_depth": 10, "min_samples_leaf": 40},
        {"n_estimators": 120, "max_depth": 18, "min_samples_leaf": 20},
    ]

    candidate_trials = []
    candidate_fold_scores = []
    candidate_oof = {}

    for method, settings_list in [
        ("CART", cart_settings),
        ("Logistic regression", logistic_settings),
        ("Random forest", forest_settings),
    ]:
        for feature_name, features in FEATURE_SETS.items():
            for settings in settings_list:
                oof = pd.Series(np.nan, index=development.index, dtype="float64")

                for fold_number in range(1, 6):
                    training = development.loc[folds.ne(fold_number)]
                    validation = development.loc[folds.eq(fold_number)]
                    normalized_weights = (
                        training["WEIGHT"] / training["WEIGHT"].mean()
                    )

                    model = make_pipeline(
                        method,
                        features,
                        NUMERIC[feature_name],
                        settings,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(
                            training[features],
                            training["injury_involved"],
                            classifier__sample_weight=normalized_weights,
                        )

                    probability = model.predict_proba(validation[features])[:, 1]
                    oof.loc[validation.index] = probability
                    candidate_fold_scores.append({
                        "method": method,
                        "features": feature_name,
                        "settings": repr(settings),
                        "fold": fold_number,
                        "n": len(validation),
                        "weighted_brier": weighted_brier(
                            validation["injury_involved"],
                            probability,
                            validation["WEIGHT"],
                        ),
                    })

                pooled_brier = weighted_brier(
                    development["injury_involved"],
                    oof,
                    development["WEIGHT"],
                )
                candidate_trials.append({
                    "method": method,
                    "features": feature_name,
                    "settings": repr(settings),
                    "pooled_weighted_brier": pooled_brier,
                })
                candidate_oof[f"{method}__{feature_name}__{repr(settings)}"] = oof

                print(
                    method,
                    feature_name,
                    repr(settings),
                    f"{pooled_brier:.6f}",
                    flush=True,
                )

    candidate_table = pd.DataFrame(candidate_trials).sort_values(
        ["pooled_weighted_brier", "method", "features", "settings"],
        kind="stable",
    ).reset_index(drop=True)
    candidate_fold_table = pd.DataFrame(candidate_fold_scores)
    candidate_fold_table = pd.DataFrame(candidate_fold_scores)

    selected_rows = (
        candidate_table.sort_values("pooled_weighted_brier")
        .drop_duplicates(["method", "features"])
        .sort_values(["method", "features"])
        .reset_index(drop=True)
    )

    secondary_rows = []
    calibration_rows = []
    for row in selected_rows.itertuples(index=False):
        label = f"{row.method}__{row.features}__{row.settings}"
        probability = candidate_oof[label]
        outcome = development["injury_involved"]
        weights = development["WEIGHT"]
        prediction = probability >= THRESHOLD
        weighted_confusion = confusion_matrix(
            outcome,
            prediction,
            labels=[0, 1],
            sample_weight=weights,
        )
        secondary_rows.append({
            "method": row.method,
            "features": row.features,
            "settings": row.settings,
            "pooled_weighted_brier": row.pooled_weighted_brier,
            "weighted_roc_auc": roc_auc_score(
                outcome, probability, sample_weight=weights
            ),
            "weighted_average_precision": average_precision_score(
                outcome, probability, sample_weight=weights
            ),
            "weighted_precision": precision_score(
                outcome, prediction, sample_weight=weights, zero_division=0
            ),
            "weighted_recall": recall_score(
                outcome, prediction, sample_weight=weights, zero_division=0
            ),
            "weighted_f1": f1_score(
                outcome, prediction, sample_weight=weights, zero_division=0
            ),
            "unweighted_brier": brier_score_loss(outcome, probability),
            "tn": float(weighted_confusion[0, 0]),
            "fp": float(weighted_confusion[0, 1]),
            "fn": float(weighted_confusion[1, 0]),
            "tp": float(weighted_confusion[1, 1]),
        })

        frame = pd.DataFrame({
            "method": row.method,
            "features": row.features,
            "outcome": outcome,
            "probability": probability,
            "weight": weights,
        })
        frame["bin"] = np.minimum((frame["probability"] * 10).astype(int), 9)
        for bin_number, group in frame.groupby("bin", sort=True):
            calibration_rows.append({
                "method": row.method,
                "features": row.features,
                "bin": bin_number,
                "n": len(group),
                "weight_sum": group["weight"].sum(),
                "mean_prediction": np.average(
                    group["probability"], weights=group["weight"]
                ),
                "observed_injury_rate": np.average(
                    group["outcome"], weights=group["weight"]
                ),
            })

    selected_secondary = pd.DataFrame(secondary_rows)
    selected_calibration = pd.DataFrame(calibration_rows)

    selected_fold_labels = set(
        (row.method, row.features, row.settings)
        for row in selected_rows.itertuples(index=False)
    )
    selected_fold_scores = candidate_fold_table[
        candidate_fold_table.apply(
            lambda row: (row["method"], row["features"], row["settings"])
            in selected_fold_labels,
            axis=1,
        )
    ].sort_values(["method", "features", "fold"])

    information_comparison = (
        selected_rows
        .pivot(index="method", columns="features", values="pooled_weighted_brier")
        [["context", "combined"]]
    )
    information_comparison["context_minus_combined"] = (
        information_comparison["context"]
        - information_comparison["combined"]
    )

    freeze_plan = {
        "test_access_during_search": False,
        "reserved_test_rows": len(crashes) - len(development),
        "overall_development_winner": candidate_table.iloc[0].to_dict(),
        "selected_by_method_feature_set": selected_rows.to_dict(orient="records"),
        "information_comparison": information_comparison.reset_index().to_dict(
            orient="records"
        ),
        "primary_metric": "pooled_survey_weighted_brier",
        "threshold": THRESHOLD,
        "seed": SEED,
        "required_before_test": [
            "Freeze method",
            "Freeze feature set",
            "Freeze hyperparameters",
            "Freeze preprocessing",
            "Freeze primary and secondary metrics",
            "Approve final-test evaluation code",
        ],
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    candidate_table.to_csv(
        RESULTS / "development_candidate_scores.csv",
        index=False,
    )
    candidate_fold_table.to_csv(
        RESULTS / "development_candidate_fold_scores.csv",
        index=False,
    )
    selected_secondary.to_csv(
        RESULTS / "development_selected_scores.csv",
        index=False,
    )
    selected_fold_scores.to_csv(
        RESULTS / "development_fold_scores.csv",
        index=False,
    )
    selected_calibration.to_csv(
        RESULTS / "development_calibration.csv",
        index=False,
    )

    oof_output = development[
        ["injury_involved", "WEIGHT", "PSU", "PSU_VAR", "PSUSTRAT"]
    ].join(folds.rename("fold"))
    for label, probability in candidate_oof.items():
        oof_output[label] = probability
    oof_output.to_csv(
        RESULTS / "development_candidate_oof.csv.gz",
        compression="gzip",
        index_label="CASENUM",
    )

    (RESULTS / "development_freeze_plan.json").write_text(
        json.dumps(freeze_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(candidate_table)
    print(selected_secondary)
    print(information_comparison)
    print(json.dumps(freeze_plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
