"""Check that the performance gate rejects incomplete or regressing evidence."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_queue_qps import compare, read_log


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def log(self, name, qps=100, *, ring=False, spill=0, wraps=1, recall=.95, threads=72):
        path = self.root / name
        text = (f"topk: 100\nbatch_q: 4\nquery_chunk: 1000\ncpu_threads: {threads}\n"
                "vanguard_size: 5\ncpu_secondary_filter: true\ncross_query_reuse: true\n"
                "rounds: 5\nwarmup_rounds: 1\nprofile: false\ncandidate_stats: false\n"
                "gpu_filter_timing: false\nnprobes: 50\n[Recall] mode=distance-aware\n")
        if ring:
            text += "queue_capacity: 4096\nqueue_spill_capacity: 1024\n"
        values = qps if isinstance(qps, list) else [qps] * 5
        for round_, value in enumerate(values, 1):
            text += f"[Run] round={round_} nprobe=50 QPS={value} recall=deferred\n"
            if ring:
                text += (f"[QueueStats] round={round_} nprobe=50 submitted=8192 "
                         f"spilled={spill} wraps={wraps}\n")
        text += f"[RecallPoint] nprobe=50 source_round=1 recall={recall}\n"
        path.write_text(text)
        return read_log(path)

    def check_status(self, after, expected, **kwargs):
        before = [self.log("a1"), self.log("a2")]
        self.assertEqual(compare(before, after, **kwargs)[0]["status"], expected)

    def test_faster_and_linear_comparisons(self):
        self.check_status([self.log("b1", 110, ring=True), self.log("b2", 110, ring=True)], "PASS")
        self.check_status([self.log("b1", 110), self.log("b2", 110)], "PASS", ring_after=False)

    def test_clear_regression(self):
        self.check_status([self.log("b1", 90, ring=True), self.log("b2", 90, ring=True)], "FAIL")

    def test_statistical_noise(self):
        before = [self.log("a1", [98, 99, 100, 101, 102]), self.log("a2", [98, 99, 100, 101, 102])]
        after = [self.log("b1", [97.9, 98.9, 99.9, 100.9, 101.9], ring=True),
                 self.log("b2", [97.9, 98.9, 99.9, 100.9, 101.9], ring=True)]
        self.assertEqual(compare(before, after)[0]["status"], "WITHIN_NOISE")

    def test_unstable_baseline(self):
        before = [self.log("a1", 90), self.log("a2", 110)]
        after = [self.log("b1", 120, ring=True), self.log("b2", 120, ring=True)]
        self.assertEqual(compare(before, after)[0]["status"], "UNSTABLE_BASELINE")

    def test_spill_and_wrap_conditions(self):
        self.check_status([self.log("b1", ring=True, spill=1), self.log("b2", ring=True)],
                          "SPILL_DETECTED")
        self.check_status([self.log("b1", ring=True, wraps=0), self.log("b2", ring=True)],
                          "MISSING_WRAP_COVERAGE", require_wrap=True)

    def test_recall_regression(self):
        self.check_status([self.log("b1", ring=True, recall=.94), self.log("b2", ring=True)],
                          "RECALL_REGRESSION")

    def test_missing_queue_stats_and_config_mismatch(self):
        before = [self.log("a1"), self.log("a2")]
        with self.assertRaises(ValueError):
            compare(before, [self.log("b1"), self.log("b2")])
        with self.assertRaises(ValueError):
            compare(before, [self.log("b1", ring=True, threads=16), self.log("b2", ring=True)])
        incomplete = self.log("b1", ring=True)
        del incomplete["config"]["batch_q"]
        with self.assertRaises(ValueError):
            compare(before, [incomplete, self.log("b2", ring=True)])

    def test_missing_or_duplicate_rounds(self):
        with self.assertRaises(ValueError):
            self.log("a1", [100, 100, 100, 100])
        entry = self.log("b1", ring=True)
        path = Path(entry["path"])
        path.write_text(path.read_text().replace("[QueueStats] round=1", "[QueueStats] round=5"))
        before = [self.log("a1"), self.log("a2")]
        with self.assertRaises(ValueError):
            compare(before, [read_log(path), self.log("b2", ring=True)])


if __name__ == "__main__":
    unittest.main()
