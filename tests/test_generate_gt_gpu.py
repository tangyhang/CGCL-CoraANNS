"""Independent FP64 brute-force oracle; CPU by default, also runnable on CUDA."""

import contextlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "data"))
import generate_gt_gpu as gt


DEVICE = os.environ.get("GT_TEST_DEVICE", "cpu")


def oracle(base, queries, k):
    ids, distances = [], []
    for q in queries:
        delta = base.astype(np.float64) - q.astype(np.float64)
        d = np.einsum("ij,ij->i", delta, delta)
        order = np.lexsort((np.arange(len(base)), d))[:k]
        ids.append(order)
        distances.append(d[order])
    return np.array(ids), np.array(distances)


def write_fbin(path, data):
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", *data.shape))
        np.asarray(data, dtype="<f4").tofile(stream)


class ExhaustiveGTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gt.torch.set_num_threads(2)

    def check_search(self, base, queries, k, **options):
        ids, distances, timing = gt.exhaustive_l2(
            base, queries, k=k, device=DEVICE, progress=False, **options)
        expected_ids, expected_d = oracle(base, queries, k)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(distances, expected_d, rtol=2e-14, atol=1e-14)
        self.assertEqual(timing["exhaustive_pairs"], len(base) * len(queries))
        self.assertTrue(all(len(np.unique(row)) == k for row in ids))
        return ids, distances

    def test_random_cross_block_top200(self):
        rng = np.random.default_rng(12)
        base = rng.normal(size=(913, 37)).astype(np.float32)
        queries = rng.normal(size=(11, 37)).astype(np.float32)
        for block, batch, refinement in [(257, 4, 53), (913, 11, 2048), (41, 3, 17)]:
            with self.subTest(block=block, batch=batch, refinement=refinement):
                self.check_search(base, queries, 200, base_block=block,
                                  query_batch=batch, refine_batch=refinement)

    def test_ties_larger_than_any_shortlist(self):
        base = np.zeros((731, 8), dtype=np.float32)
        base[:17] = 3
        queries = np.zeros((3, 8), dtype=np.float32)
        self.check_search(base, queries, 200, base_block=211,
                          query_batch=2, refine_batch=19)

    def test_cancellation_requires_refining_all_pairs(self):
        rng = np.random.default_rng(35)
        base = (1e5 + rng.normal(size=(109, 15))).astype(np.float32)
        queries = (1e5 + rng.normal(size=(5, 15))).astype(np.float32)
        self.check_search(base, queries, 27, base_block=37,
                          query_batch=2, refine_batch=13)

    def test_subnormal_products(self):
        base = (np.arange(24).reshape(8, 3) * 1e-23).astype(np.float32)
        queries = np.zeros((2, 3), dtype=np.float32)
        self.check_search(base, queries, 5, base_block=3, refine_batch=4)

    def test_single_vector(self):
        self.check_search(np.array([[2, 4]], dtype=np.float32),
                          np.array([[0, 3]], dtype=np.float32), 1)

    def test_nonfinite_base_rejected_even_outside_topk(self):
        for bad in [np.nan, np.inf]:
            base = np.ones((30, 4), dtype=np.float32)
            base[-1, 0] = bad
            with self.assertRaisesRegex(ValueError, "base norm"):
                gt.exhaustive_l2(base, base[:2], k=3, device=DEVICE,
                                 base_block=11, progress=False)

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            gt.exhaustive_l2(np.zeros((2, 4)), np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "topk"):
            gt.exhaustive_l2(np.zeros((2, 4)), np.zeros((2, 4)), k=3)
        with self.assertRaisesRegex(ValueError, "NaN"):
            gt.exhaustive_l2(np.zeros((2, 4)), np.full((1, 4), np.nan), k=1,
                             device=DEVICE)

    def test_format_roundtrip_and_no_overwrite(self):
        rng = np.random.default_rng(7)
        base = rng.normal(size=(503, 19)).astype(np.float32)
        queries = rng.normal(size=(7, 19)).astype(np.float32)
        old, _ = oracle(base, queries, 100)
        expected, _ = oracle(base, queries, 200)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_fbin(path / "base.fbin", base)
            write_fbin(path / "query.fbin", queries)
            gt.write_ivecs(path / "old.ivecs", old)
            args = ["--base", str(path / "base.fbin"), "--query", str(path / "query.fbin"),
                    "--output", str(path / "new.ivecs"), "--existing-gt", str(path / "old.ivecs"),
                    "--device", DEVICE, "--base-block", "201", "--query-batch", "3",
                    "--refine-batch", "31", "--threads", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                gt.main(args)
            actual = gt.read_ivecs(path / "new.ivecs", len(queries), len(base))
            np.testing.assert_array_equal(actual, expected)
            raw = np.fromfile(path / "new.ivecs", dtype="<i4").reshape(7, 201)
            np.testing.assert_array_equal(raw[:, 0], 200)
            metadata = json.loads((path / "new.ivecs.json").read_text())
            self.assertEqual(metadata["compatibility"]["mean_id_overlap"], 1.0)
            self.assertEqual(metadata["compatibility"]["identical_prefix_queries"], 7)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                gt.main(args)
            with self.assertRaises(FileExistsError):
                gt.write_ivecs(path / "new.ivecs", old)
            np.testing.assert_array_equal(gt.read_ivecs(path / "new.ivecs", 7, 503), expected)

    def test_malformed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            file = path / "base.fbin"
            file.write_bytes(struct.pack("<II", 3, 4) + bytes(4))
            with self.assertRaisesRegex(ValueError, "file size"):
                gt.read_fbin(file)
            file.write_bytes(bytes(2))
            with self.assertRaisesRegex(ValueError, "header"):
                gt.read_fbin(file)
            old = path / "old.ivecs"
            np.array([2, 0, 0], dtype="<i4").tofile(old)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                gt.read_ivecs(old, 1, 10)
            np.array([2, 0, 10], dtype="<i4").tofile(old)
            with self.assertRaisesRegex(ValueError, "range"):
                gt.read_ivecs(old, 1, 10)

    def test_existing_gt_reports_inversions_without_copying_them(self):
        base = np.array([[0], [1], [2], [3]], dtype=np.float32)
        queries = np.array([[0]], dtype=np.float32)
        ids, distances = oracle(base, queries, 4)
        report = gt.compare_existing(base, queries, ids, distances, np.array([[1, 0]]))
        self.assertEqual(report["old_distance_inversions"], 1)
        self.assertEqual(report["identical_topk_set_queries"], 1)
        self.assertEqual(report["identical_prefix_queries"], 0)
        np.testing.assert_array_equal(ids, [[0, 1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
