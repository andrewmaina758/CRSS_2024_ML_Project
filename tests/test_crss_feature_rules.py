"""Focused tests for audited CRSS target and vehicle rules."""

from __future__ import annotations

import math
import unittest

from scripts.crss_feature_rules import (
    BinaryTargetClassification,
    DEFINITE_LARGE_TRUCK_BODY_TYPES,
    VALID_TOW_VEH_CODES,
    classify_imputed_sensitivity_target,
    classify_large_truck,
    classify_primary_raw_target,
)


class PrimaryRawTargetTests(unittest.TestCase):
    def test_primary_target_mapping(self) -> None:
        expected = {
            0: BinaryTargetClassification(True, 0),
            1: BinaryTargetClassification(True, 1),
            2: BinaryTargetClassification(True, 1),
            3: BinaryTargetClassification(True, 1),
            4: BinaryTargetClassification(True, 1),
            5: BinaryTargetClassification(True, 1),
            6: BinaryTargetClassification(False, None),
            8: BinaryTargetClassification(False, None),
            9: BinaryTargetClassification(False, None),
        }

        for code, result in expected.items():
            with self.subTest(code=code):
                self.assertEqual(classify_primary_raw_target(code), result)

    def test_primary_target_rejects_missing_or_unknown_codes(self) -> None:
        for value in (None, math.nan, 7, 10, 1.5, "1", True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    classify_primary_raw_target(value)


class ImputedSensitivityTargetTests(unittest.TestCase):
    def test_imputed_target_mapping(self) -> None:
        expected = {
            0: BinaryTargetClassification(True, 0),
            1: BinaryTargetClassification(True, 1),
            2: BinaryTargetClassification(True, 1),
            3: BinaryTargetClassification(True, 1),
            4: BinaryTargetClassification(True, 1),
            5: BinaryTargetClassification(True, 1),
            6: BinaryTargetClassification(False, None),
            8: BinaryTargetClassification(False, None),
        }

        for code, result in expected.items():
            with self.subTest(code=code):
                self.assertEqual(
                    classify_imputed_sensitivity_target(code),
                    result,
                )

    def test_imputed_target_rejects_unresolved_or_invalid_codes(self) -> None:
        for value in (None, math.nan, 7, 9, 10, 2.5, "2", False):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    classify_imputed_sensitivity_target(value)


class LargeTruckTests(unittest.TestCase):
    def test_documented_medium_heavy_body_types_are_yes(self) -> None:
        for body_typ in DEFINITE_LARGE_TRUCK_BODY_TYPES:
            for tow_veh in (0, 1, 4, 7, 9, None):
                with self.subTest(body_typ=body_typ, tow_veh=tow_veh):
                    self.assertEqual(
                        classify_large_truck(body_typ, tow_veh),
                        "yes",
                    )

    def test_body_type_79_is_unknown_for_every_towing_code(self) -> None:
        for tow_veh in (*sorted(VALID_TOW_VEH_CODES), None):
            with self.subTest(tow_veh=tow_veh):
                self.assertEqual(classify_large_truck(79, tow_veh), "unknown")

    def test_towing_alone_never_makes_a_large_truck(self) -> None:
        for body_typ in (22, 34, 39, 48, 65, 73):
            for tow_veh in (1, 2, 3, 4, 5, 6, 7):
                with self.subTest(body_typ=body_typ, tow_veh=tow_veh):
                    self.assertEqual(classify_large_truck(body_typ, tow_veh), "no")

    def test_unreported_unknown_or_missing_body_type_is_unknown(self) -> None:
        for body_typ in (None, math.nan, 79, 98, 99):
            with self.subTest(body_typ=body_typ):
                self.assertEqual(classify_large_truck(body_typ, 0), "unknown")

    def test_invalid_towing_code_is_rejected(self) -> None:
        for tow_veh in (-1, 8, 10, 1.5, "1", True):
            with self.subTest(tow_veh=tow_veh):
                with self.assertRaises(ValueError):
                    classify_large_truck(60, tow_veh)

    def test_invalid_body_type_is_rejected(self) -> None:
        for body_typ in (0, 100, -1, 60.5, "60", True):
            with self.subTest(body_typ=body_typ):
                with self.assertRaises(ValueError):
                    classify_large_truck(body_typ, 0)


if __name__ == "__main__":
    unittest.main()
