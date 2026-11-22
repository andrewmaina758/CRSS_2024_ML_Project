import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.crss_provenance import (
    build_metadata,
    build_split_manifest,
    persist_or_validate_manifest,
)


class SplitManifestTests(unittest.TestCase):
    def setUp(self):
        case_ids = range(1000, 1100)
        self.frame = pd.DataFrame(
            {
                "target": [value % 2 for value in case_ids],
                "PSU": [value % 10 for value in case_ids],
            },
            index=pd.Index(case_ids, name="CASENUM"),
        )

    def test_manifest_is_deterministic_complete_and_grouped(self):
        first = build_split_manifest(
            self.frame,
            target_column="target",
            group_column="PSU",
            seed=678,
        )
        second = build_split_manifest(
            self.frame,
            target_column="target",
            group_column="PSU",
            seed=678,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["CASENUM"]), set(self.frame.index))
        self.assertEqual(first["partition"].eq("final_test").sum(), 20)

        joined = first.merge(
            self.frame["PSU"].rename_axis("CASENUM").reset_index(),
            on="CASENUM",
        )
        grouped = joined[joined["partition"].eq("development")].groupby("PSU")
        self.assertTrue(grouped["psu_group_fold"].nunique().eq(1).all())

    def test_persistence_is_idempotent_and_detects_changed_hash(self):
        manifest = build_split_manifest(
            self.frame,
            target_column="target",
            group_column="PSU",
            seed=678,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            source = root / "source.zip"
            self.frame.to_csv(dataset)
            source.write_bytes(b"source")
            metadata = build_metadata(
                dataset_path=dataset,
                source_path=source,
                manifest=manifest,
                seed=678,
                test_size=0.2,
                target_definition={"column": "target"},
                feature_sets={"context": ["PSU"]},
                schema={"PSU": "int64"},
            )
            csv_path = root / "manifest.csv"
            json_path = root / "manifest.json"
            persist_or_validate_manifest(
                manifest, metadata, csv_path=csv_path, json_path=json_path
            )
            loaded, loaded_metadata = persist_or_validate_manifest(
                manifest, metadata, csv_path=csv_path, json_path=json_path
            )
            self.assertEqual(len(loaded), len(manifest))
            self.assertEqual(loaded_metadata["dataset_sha256"], metadata["dataset_sha256"])

            changed = json.loads(json.dumps(metadata))
            changed["dataset_sha256"] = "DIFFERENT"
            with self.assertRaises(ValueError):
                persist_or_validate_manifest(
                    manifest, changed, csv_path=csv_path, json_path=json_path
                )


if __name__ == "__main__":
    unittest.main()
