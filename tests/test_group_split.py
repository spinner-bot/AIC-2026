import unittest

import pandas as pd

from src.capr.group_split import build_groups, split_train_val


class GroupSplitTest(unittest.TestCase):
    def test_cross_class_duplicates_never_cross_folds(self) -> None:
        rows = []
        for class_idx in range(2):
            for index in range(12):
                rows.append(
                    {
                        "class_idx": class_idx,
                        "file_hash": f"{class_idx}-{index}",
                        "phash": f"{class_idx * 1000 + index * 100:016x}",
                    }
                )
        rows[12]["file_hash"] = rows[0]["file_hash"]
        rows[12]["phash"] = rows[0]["phash"]
        manifest = pd.DataFrame(rows)

        grouped = build_groups(manifest, near_dup_hamming=0)
        self.assertEqual(grouped.loc[0, "group_id"], grouped.loc[12, "group_id"])

        split = split_train_val(grouped, seed=42)
        self.assertTrue((split.groupby("group_id")["fold"].nunique() == 1).all())

    def test_near_duplicate_is_global(self) -> None:
        manifest = pd.DataFrame(
            [
                {"class_idx": 0, "file_hash": "a", "phash": "0000000000000000"},
                {"class_idx": 1, "file_hash": "b", "phash": "0000000000000001"},
            ]
        )
        grouped = build_groups(manifest, near_dup_hamming=1)
        self.assertEqual(grouped.loc[0, "group_id"], grouped.loc[1, "group_id"])


if __name__ == "__main__":
    unittest.main()
