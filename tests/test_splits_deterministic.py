from __future__ import annotations

import random
import unittest

from scripts.split_dataset import split_task_records


class SplitDeterministicTest(unittest.TestCase):
    def _records(self) -> list[dict]:
        rows: list[dict] = []
        for session_idx in range(20):
            session_id = f"session-{session_idx:03d}"
            for sample_idx in range(3):
                rows.append(
                    {
                        "sessionId": session_id,
                        "sampleId": f"{session_id}-sample-{sample_idx}",
                        "sourceLine": sample_idx + 1,
                        "text": "request",
                        "label": "VALID",
                    }
                )
        return rows

    def test_deterministic_assignments_independent_of_input_order(self) -> None:
        original = self._records()
        shuffled = list(original)
        random.Random(7).shuffle(shuffled)

        split_a = split_task_records(
            original, seed="seed-A", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )
        split_b = split_task_records(
            shuffled, seed="seed-A", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )

        for split_name in ("train", "val", "test"):
            ids_a = {record["sampleId"] for record in split_a[split_name]}
            ids_b = {record["sampleId"] for record in split_b[split_name]}
            self.assertEqual(ids_a, ids_b, f"split mismatch for {split_name}")

    def test_session_records_stay_in_one_split(self) -> None:
        records = self._records()
        split_map = split_task_records(
            records, seed="seed-B", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )

        session_to_split: dict[str, str] = {}
        for split_name, items in split_map.items():
            for record in items:
                session_id = record["sessionId"]
                previous = session_to_split.get(session_id)
                if previous is None:
                    session_to_split[session_id] = split_name
                else:
                    self.assertEqual(
                        previous, split_name, f"session {session_id} leaked across splits"
                    )


if __name__ == "__main__":
    unittest.main()
