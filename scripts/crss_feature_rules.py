"""Audited CRSS 2024 target and large-truck feature definitions.

The code meanings in this module were checked against the paired ``*NAME``
columns in ``data/CRSS2024_three_tables.zip`` and the NHTSA *2024 FARS/CRSS
Coding and Validation Manual* (DOT HS 813 798).

The primary target deliberately uses reported ``MAX_SEV``.  The imputed
``MAXSEV_IM`` definition is a sensitivity target, not an interchangeable
replacement.  Large-truck status is conservative: ``BODY_TYP == 79`` means
that light versus medium/heavy truck is unresolved, and ``TOW_VEH`` describes
trailing units rather than vehicle weight class.  Neither is positive evidence
of a large truck by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal


@dataclass(frozen=True, slots=True)
class BinaryTargetClassification:
    """Eligibility and binary label for one crash.

    ``label`` is ``None`` whenever the crash is outside the corresponding
    target population.
    """

    eligible: bool
    label: int | None


LargeTruckState = Literal["no", "unknown", "yes"]

NO_REPORTED_INJURY_CODE = 0
REPORTED_INJURY_CODES = frozenset({1, 2, 3, 4, 5})

# MAX_SEV adds code 9 (unknown/not reported); MAXSEV_IM resolves those records
# in the supplied 2024 release and therefore has no observed code 9.
RAW_MAX_SEV_CODES = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9})
IMPUTED_MAX_SEV_CODES = frozenset({0, 1, 2, 3, 4, 5, 6, 8})

# These codes explicitly identify a medium/heavy truck body class in the 2024
# coding system.  Codes 71 and 72 are valid definitions even though neither is
# present in the supplied vehicle table.
DEFINITE_LARGE_TRUCK_BODY_TYPES = frozenset(
    {60, 61, 62, 63, 64, 66, 67, 71, 72, 78}
)

# 79 cannot resolve light versus medium/heavy.  Codes 98 and 99 are respectively
# not reported and unknown.  A missing BODY_TYP has the same unknown state.
UNKNOWN_LARGE_TRUCK_BODY_TYPES = frozenset({79, 98, 99})

VALID_TOW_VEH_CODES = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 9})


def classify_primary_raw_target(max_sev: object) -> BinaryTargetClassification:
    """Classify the primary reported-injury target from ``MAX_SEV``.

    Codes 0 through 5 are eligible.  Code 0 is no apparent injury, while codes
    1 through 5 are reported injury severities.  Codes 6 (died before crash),
    8 (no person involved), and 9 (unknown/not reported) are excluded.

    Missing, non-integral, and undocumented codes raise ``ValueError`` so data
    problems cannot be silently converted into exclusions.
    """

    code = _required_integer_code(max_sev, field="MAX_SEV")
    return _classify_target(code, valid_codes=RAW_MAX_SEV_CODES, field="MAX_SEV")


def classify_imputed_sensitivity_target(
    maxsev_im: object,
) -> BinaryTargetClassification:
    """Classify the sensitivity target from imputed ``MAXSEV_IM``.

    Codes 0 through 5 use the same binary mapping as the primary target.  Codes
    6 and 8 remain ineligible.  The supplied 2024 imputed field contains no
    code 9; encountering it (or another undocumented code) raises ``ValueError``
    instead of silently treating an incomplete imputation as valid input.
    """

    code = _required_integer_code(maxsev_im, field="MAXSEV_IM")
    return _classify_target(
        code,
        valid_codes=IMPUTED_MAX_SEV_CODES,
        field="MAXSEV_IM",
    )


def classify_large_truck(body_typ: object, tow_veh: object = None) -> LargeTruckState:
    """Return ``"yes"``, ``"no"``, or ``"unknown"`` for one vehicle.

    ``BODY_TYP`` codes 60-64, 66, 67, 71, 72, and 78 are definitive
    medium/heavy truck evidence.  Code 79 is always ``"unknown"`` because its
    documented meaning is unknown truck type (light/medium/heavy).  Codes 98,
    99, and missing body type are also ``"unknown"``.  Other body types are
    ``"no"`` for this feature; notably, 65 is a medium/heavy vehicle-based
    motor home and 73 is a camper/motor home with unknown GVWR.

    ``TOW_VEH`` is validated when present, but it never changes the result.
    Its 2024 codes describe the number or kind of trailing unit, not the
    towing vehicle's GVWR.  Thus a light/unknown truck towing a trailer is not
    promoted to a definitive large truck.
    """

    towing_code = _optional_integer_code(tow_veh, field="TOW_VEH")
    if towing_code is not None and towing_code not in VALID_TOW_VEH_CODES:
        raise ValueError(
            f"Unexpected TOW_VEH code {towing_code!r}; "
            f"expected one of {sorted(VALID_TOW_VEH_CODES)}"
        )

    body_code = _optional_integer_code(body_typ, field="BODY_TYP")
    if body_code is None or body_code in UNKNOWN_LARGE_TRUCK_BODY_TYPES:
        return "unknown"

    if not 1 <= body_code <= 99:
        raise ValueError(f"Unexpected BODY_TYP code {body_code!r}; expected 1-99")

    if body_code in DEFINITE_LARGE_TRUCK_BODY_TYPES:
        return "yes"

    return "no"


def _classify_target(
    code: int,
    *,
    valid_codes: frozenset[int],
    field: str,
) -> BinaryTargetClassification:
    if code not in valid_codes:
        raise ValueError(
            f"Unexpected {field} code {code!r}; expected one of {sorted(valid_codes)}"
        )

    if code == NO_REPORTED_INJURY_CODE:
        return BinaryTargetClassification(eligible=True, label=0)
    if code in REPORTED_INJURY_CODES:
        return BinaryTargetClassification(eligible=True, label=1)
    return BinaryTargetClassification(eligible=False, label=None)


def _required_integer_code(value: object, *, field: str) -> int:
    code = _optional_integer_code(value, field=field)
    if code is None:
        raise ValueError(f"{field} is missing")
    return code


def _optional_integer_code(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer code, not bool")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric_value = float(value)
        if math.isnan(numeric_value):
            return None
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            return int(numeric_value)
    raise ValueError(f"{field} must be an integer code; received {value!r}")


__all__ = [
    "BinaryTargetClassification",
    "DEFINITE_LARGE_TRUCK_BODY_TYPES",
    "IMPUTED_MAX_SEV_CODES",
    "RAW_MAX_SEV_CODES",
    "UNKNOWN_LARGE_TRUCK_BODY_TYPES",
    "VALID_TOW_VEH_CODES",
    "classify_imputed_sensitivity_target",
    "classify_large_truck",
    "classify_primary_raw_target",
]
