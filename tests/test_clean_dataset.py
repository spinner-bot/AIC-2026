import argparse
import hashlib
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.clean_dataset import clean_dataset


def _jpeg(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 12), color).save(buffer, format="JPEG")
    return buffer.getvalue()


class CleanDatasetTest(unittest.TestCase):
    def test_end_to_end_keeps_sources_and_groups_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_zip = root / "train.zip"
            test_zip = root / "test.zip"
            duplicate = _jpeg((255, 0, 0))
            with zipfile.ZipFile(train_zip, "w") as archive:
                archive.writestr("0000/a.jpg", duplicate)
                archive.writestr("0000/b.jpg", _jpeg((0, 255, 0)))
                archive.writestr("0001/c.jpg", duplicate)
                archive.writestr("0001/d.jpg", _jpeg((0, 0, 255)))
            with zipfile.ZipFile(test_zip, "w") as archive:
                archive.writestr("test.jpg", _jpeg((10, 20, 30)))

            source_hashes = {
                train_zip: hashlib.sha256(train_zip.read_bytes()).hexdigest(),
                test_zip: hashlib.sha256(test_zip.read_bytes()).hexdigest(),
            }
            args = argparse.Namespace(
                train_zip=train_zip,
                test_zip=test_zip,
                output_dir=root / "cleaned",
                expected_classes=2,
                expected_train=4,
                expected_test=1,
                phash_size=8,
                near_dup_hamming=0,
                val_ratio=0.1,
                min_samples_for_split=10,
                oof_folds=3,
                head_quantile=0.33,
                tail_quantile=0.33,
                seed=42,
                progress_every=100,
            )

            summary = clean_dataset(args)
            self.assertEqual(summary["train"]["accepted"], 4)
            self.assertEqual(summary["test"]["accepted"], 1)
            for source, digest in source_hashes.items():
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)

            split = pd.read_csv(root / "cleaned" / "reports" / "split.csv")
            duplicate_rows = split[split["file_hash"] == hashlib.sha256(duplicate).hexdigest()]
            self.assertEqual(duplicate_rows["group_id"].nunique(), 1)
            self.assertEqual(duplicate_rows["fold"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
