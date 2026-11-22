"""Reproducible split manifests and provenance for the CRSS CART workflow.

The manifest is deliberately separate from model output.  It records which crash
belongs to development or the sealed final-test partition and fixes all
development fold assignments before any model is selected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


def file_sha256(path: str | Path) -> str:
    """Return the uppercase SHA-256 digest of *path*."""

    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _validate_binary_target(target: pd.Series) -> None:
    if target.isna().any():
        raise ValueError("The split target contains missing values")
    values = set(pd.unique(target))
    if values != {0, 1}:
        raise ValueError(f"Expected binary target values {{0, 1}}; received {values}")


def build_split_manifest(
    frame: pd.DataFrame,
    *,
    target_column: str,
    group_column: str,
    seed: int,
    test_size: float = 0.20,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Create fixed primary and PSU-grouped development folds.

    ``frame`` must be indexed by unique ``CASENUM`` values.  ``outer_fold`` is
    the ordinary stratified fold used for primary nested evaluation.
    ``psu_group_fold`` keeps every value of ``group_column`` in one fold and is
    used only as a geographic/design sensitivity analysis.
    """

    if not frame.index.is_unique:
        raise ValueError("CASENUM index must be unique")
    if frame.index.hasnans:
        raise ValueError("CASENUM index contains a missing value")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between zero and one")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if group_column not in frame:
        raise KeyError(f"Missing group column: {group_column}")
    if frame[group_column].isna().any():
        raise ValueError(f"{group_column} contains missing values")

    target = frame[target_column]
    _validate_binary_target(target)

    development_ids, final_test_ids = train_test_split(
        frame.index.to_numpy(),
        test_size=test_size,
        stratify=target.to_numpy(),
        random_state=seed,
    )
    development_ids = np.sort(development_ids)
    final_test_ids = np.sort(final_test_ids)

    manifest = pd.DataFrame(index=pd.Index(frame.index, name="CASENUM"))
    manifest["partition"] = "final_test"
    manifest.loc[development_ids, "partition"] = "development"
    manifest["outer_fold"] = pd.Series(pd.NA, index=manifest.index, dtype="Int8")
    manifest["psu_group_fold"] = pd.Series(
        pd.NA, index=manifest.index, dtype="Int8"
    )

    development = frame.loc[development_ids]
    row_splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    for fold, (_, validation_positions) in enumerate(
        row_splitter.split(development, development[target_column]), start=1
    ):
        validation_ids = development.index[validation_positions]
        manifest.loc[validation_ids, "outer_fold"] = fold

    group_splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    for fold, (_, validation_positions) in enumerate(
        group_splitter.split(
            development,
            development[target_column],
            groups=development[group_column],
        ),
        start=1,
    ):
        validation_ids = development.index[validation_positions]
        manifest.loc[validation_ids, "psu_group_fold"] = fold

    manifest = manifest.sort_index().reset_index()
    validate_split_manifest(
        manifest,
        expected_ids=frame.index,
        groups=frame[group_column],
        n_splits=n_splits,
    )
    return manifest


def validate_split_manifest(
    manifest: pd.DataFrame,
    *,
    expected_ids: Sequence[Any] | pd.Index,
    groups: pd.Series,
    n_splits: int = 5,
) -> None:
    """Raise when a manifest is incomplete, overlapping, or splits a PSU."""

    required = {"CASENUM", "partition", "outer_fold", "psu_group_fold"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest["CASENUM"].duplicated().any():
        raise ValueError("Manifest contains duplicate CASENUM values")
    if set(manifest["CASENUM"]) != set(expected_ids):
        raise ValueError("Manifest CASENUM values do not match the analysis cohort")
    if not manifest["partition"].isin(["development", "final_test"]).all():
        raise ValueError("Manifest contains an invalid partition")

    development = manifest["partition"].eq("development")
    final_test = ~development
    valid_folds = set(range(1, n_splits + 1))
    for column in ("outer_fold", "psu_group_fold"):
        observed = set(manifest.loc[development, column].dropna().astype(int))
        if observed != valid_folds:
            raise ValueError(f"{column} does not contain every expected fold")
        if manifest.loc[development, column].isna().any():
            raise ValueError(f"A development row is missing {column}")
        if manifest.loc[final_test, column].notna().any():
            raise ValueError(f"A final-test row has an assigned {column}")

    group_lookup = groups.rename("group").rename_axis("CASENUM").reset_index()
    group_folds = (
        manifest.loc[development, ["CASENUM", "psu_group_fold"]]
        .merge(group_lookup, on="CASENUM", validate="one_to_one")
        .groupby("group")["psu_group_fold"]
        .nunique()
    )
    if group_folds.gt(1).any():
        raise ValueError("At least one PSU is split across grouped folds")


def build_metadata(
    *,
    dataset_path: str | Path,
    source_path: str | Path,
    manifest: pd.DataFrame,
    seed: int,
    test_size: float,
    target_definition: Mapping[str, Any],
    feature_sets: Mapping[str, Sequence[str]],
    schema: Mapping[str, str],
) -> dict[str, Any]:
    """Create the JSON-serializable provenance record for a manifest."""

    canonical_features = {
        name: list(columns) for name, columns in sorted(feature_sets.items())
    }
    schema_payload = json.dumps(
        {"feature_sets": canonical_features, "schema": dict(sorted(schema.items()))},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": Path(dataset_path).as_posix(),
        "dataset_sha256": file_sha256(dataset_path),
        "source_path": Path(source_path).as_posix(),
        "source_sha256": file_sha256(source_path),
        "seed": int(seed),
        "test_size": float(test_size),
        "target_definition": dict(target_definition),
        "feature_sets": canonical_features,
        "schema": dict(sorted(schema.items())),
        "schema_sha256": sha256(schema_payload).hexdigest().upper(),
        "row_count": int(len(manifest)),
        "development_count": int(manifest["partition"].eq("development").sum()),
        "final_test_count": int(manifest["partition"].eq("final_test").sum()),
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def persist_or_validate_manifest(
    manifest: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    csv_path: str | Path,
    json_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a manifest once, then require exact provenance on every rerun."""

    csv_path = Path(csv_path)
    json_path = Path(json_path)
    if csv_path.exists() != json_path.exists():
        raise FileNotFoundError("Manifest CSV and metadata JSON must exist together")

    if csv_path.exists():
        stored_manifest = pd.read_csv(
            csv_path,
            dtype={"outer_fold": "Int8", "psu_group_fold": "Int8"},
        )
        stored_metadata = json.loads(json_path.read_text(encoding="utf-8"))
        stable_keys = (
            "dataset_sha256",
            "source_sha256",
            "seed",
            "test_size",
            "target_definition",
            "feature_sets",
            "schema",
            "schema_sha256",
            "row_count",
            "development_count",
            "final_test_count",
        )
        mismatches = [
            key for key in stable_keys if stored_metadata.get(key) != metadata.get(key)
        ]
        if mismatches:
            raise ValueError(
                "Stored manifest provenance differs for: " + ", ".join(mismatches)
            )
        comparable_columns = [
            "CASENUM",
            "partition",
            "outer_fold",
            "psu_group_fold",
        ]
        pd.testing.assert_frame_equal(
            stored_manifest[comparable_columns].reset_index(drop=True),
            manifest[comparable_columns].reset_index(drop=True),
            check_dtype=False,
        )
        return stored_manifest, stored_metadata

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest.copy(), dict(metadata)
