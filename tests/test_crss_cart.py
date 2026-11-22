"""Focused tests for the reusable CART evaluation workflow."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from scripts.crss_cart import (
    SigmoidProbabilityCalibrator,
    evaluate_cart_with_psu_sensitivity,
    evaluate_nested_cart,
    make_cart_pipeline,
)


def synthetic_crashes(
    *,
    n_groups: int = 24,
    rows_per_group: int = 8,
    seed: int = 19,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build a small, weighted binary problem with repeated PSU groups."""

    rng = np.random.default_rng(seed)
    n_rows = n_groups * rows_per_group
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    speed = rng.normal(43.0, 12.0, n_rows)
    vehicles = rng.integers(1, 5, n_rows)
    weather = rng.choice(["clear", "rain", "snow"], n_rows, p=[0.7, 0.2, 0.1])
    road_user = rng.choice(["occupant", "motorcycle", "pedestrian"], n_rows)
    group_effect = rng.normal(0.0, 0.25, n_groups)[groups]
    linear = (
        -2.0
        + 0.025 * speed
        + 0.22 * vehicles
        + 0.55 * (weather == "rain")
        + 1.0 * (road_user == "motorcycle")
        + 1.5 * (road_user == "pedestrian")
        + group_effect
    )
    probability = 1.0 / (1.0 + np.exp(-linear))
    outcome = rng.binomial(1, probability)
    # Guarantee both classes within a comfortably large set of groups.
    outcome[0] = 0
    outcome[1] = 1
    index = pd.Index(np.arange(10_000, 10_000 + n_rows), name="CASENUM")
    features = pd.DataFrame(
        {
            "speed": speed,
            "vehicles": vehicles,
            "weather": weather,
            "road_user": road_user,
        },
        index=index,
    )
    target = pd.Series(outcome, index=index, name="injury_involved")
    weights = pd.Series(rng.uniform(0.4, 3.0, n_rows), index=index, name="WEIGHT")
    psu = pd.Series(groups, index=index, name="PSU")
    return features, target, weights, psu


class NestedCartTests(unittest.TestCase):
    numeric = ["speed", "vehicles"]
    categorical = ["weather", "road_user"]
    small_grid = {
        "max_depth": [2, 4],
        "min_samples_leaf": [6],
        "ccp_alpha": [0.0],
    }

    def test_nested_evaluation_returns_complete_weighted_artifacts(self) -> None:
        features, target, weights, _ = synthetic_crashes()
        result = evaluate_nested_cart(
            features,
            target,
            weights,
            numeric_features=self.numeric,
            categorical_features=self.categorical,
            param_grid=self.small_grid,
            outer_splits=3,
            inner_splits=2,
            calibration_splits=2,
            random_state=31,
        )

        self.assertTrue(result.predictions.index.equals(features.index))
        self.assertFalse(result.predictions.isna().any().any())
        self.assertEqual(set(result.predictions["outer_fold"]), {1, 2, 3})
        for column in ("raw_probability", "calibrated_probability"):
            self.assertTrue(result.predictions[column].between(0.0, 1.0).all())

        expected_raw = brier_score_loss(
            target,
            result.predictions["raw_probability"],
            sample_weight=weights,
        )
        expected_calibrated = brier_score_loss(
            target,
            result.predictions["calibrated_probability"],
            sample_weight=weights,
        )
        self.assertAlmostEqual(result.raw_weighted_brier, expected_raw)
        self.assertAlmostEqual(result.calibrated_weighted_brier, expected_calibrated)
        self.assertEqual(len(result.fold_summary), 3)
        self.assertEqual(len(result.search_results), 3 * 2)
        self.assertEqual(
            int(result.leaf_diagnostics["n_validation"].sum()),
            len(features),
        )
        self.assertTrue(
            {
                "training_event_rate",
                "validation_event_rate",
                "raw_probability",
                "calibrated_probability",
            }.issubset(result.leaf_diagnostics.columns)
        )

    def test_psu_grouped_sensitivity_never_splits_a_group(self) -> None:
        features, target, weights, psu = synthetic_crashes(n_groups=30)
        result = evaluate_cart_with_psu_sensitivity(
            features,
            target,
            weights,
            psu,
            numeric_features=self.numeric,
            categorical_features=self.categorical,
            param_grid={
                "max_depth": [3],
                "min_samples_leaf": [6],
                "ccp_alpha": [0.0],
            },
            outer_splits=3,
            inner_splits=2,
            calibration_splits=2,
            random_state=47,
        )

        grouped_predictions = result.psu_grouped.predictions
        folds_per_group = grouped_predictions.groupby("group")["outer_fold"].nunique()
        self.assertTrue(folds_per_group.eq(1).all())
        self.assertTrue(result.psu_grouped.grouped_outer)
        self.assertFalse(result.row_wise.grouped_outer)
        self.assertEqual(
            result.comparison["outer_strategy"].tolist(),
            ["row_wise", "psu_grouped"],
        )

    def test_pipeline_factory_returns_fresh_models(self) -> None:
        first = make_cart_pipeline(
            self.numeric,
            self.categorical,
            tree_params={"max_depth": 3, "min_samples_leaf": 5},
        )
        second = make_cart_pipeline(
            self.numeric,
            self.categorical,
            tree_params={"max_depth": 3, "min_samples_leaf": 5},
        )
        self.assertIsNot(first, second)
        self.assertIsNot(first.named_steps["prepare"], second.named_steps["prepare"])
        self.assertIsNot(
            first.named_steps["classifier"],
            second.named_steps["classifier"],
        )

    def test_sigmoid_calibrator_validates_inputs(self) -> None:
        calibrator = SigmoidProbabilityCalibrator()
        with self.assertRaisesRegex(ValueError, "both binary outcome classes"):
            calibrator.fit([0.2, 0.4], [0, 0], [1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "within \[0, 1\]"):
            calibrator.fit([0.2, 1.2], [0, 1], [1.0, 1.0])

    def test_invalid_weights_and_feature_overlap_fail_early(self) -> None:
        features, target, weights, _ = synthetic_crashes()
        bad_weights = weights.copy()
        bad_weights.iloc[0] = 0.0
        with self.assertRaisesRegex(ValueError, "must all be positive"):
            evaluate_nested_cart(
                features,
                target,
                bad_weights,
                numeric_features=self.numeric,
                categorical_features=self.categorical,
                param_grid=self.small_grid,
                outer_splits=2,
                inner_splits=2,
                calibration_splits=2,
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            make_cart_pipeline(["speed"], ["speed"])


if __name__ == "__main__":
    unittest.main()
