"""Exercise result provenance and failure handling without a GPU or datasets."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RunnerIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="coraanns runner ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "bin").mkdir()
        self.data = self.root / "data with spaces"
        self.data.mkdir()
        for name in ["index", "queries", "truth", "base"]:
            (self.data / name).write_bytes(b"fixture")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({
            "dataset": "test_fixture", "nlist": 8,
            "files": dict(index="index", queries="queries", groundtruth="truth", base="base"),
            "search": dict(topk=2, rounds=5, warmup=1, threads=2, batch_q=4,
                           query_chunk=2, gpu_query_chunk=2, vanguard_size=5,
                           cpu_secondary_filter=True, cross_query_reuse=True),
            "experiments": {"overall": {"nprobes": [2, 4]}}
        }))
        # These are test fixtures, not performance measurements.
        for name in ["coraanns_search", "rabitq_cpu", "rabitq_gpu"]:
            self.binary(name)

    def binary(self, name, *, code=0, mode=None, incomplete=False):
        text = (f"#!{sys.executable}\nimport sys\n"
                f"mode = {mode!r} or ('distance-aware' if '--recall-base' in sys.argv else 'id')\n"
                "print('[Recall] mode=' + mode)\n"
                "print('nprobe\\tQPS\\trecall')\n"
                "print('2\\t100\\t0.5')\n" +
                ("" if incomplete else "print('4\\t200\\t0.75')\n") +
                f"sys.exit({code})\n")
        path = self.root / "bin" / name
        path.write_text(text)
        path.chmod(0o755)

    def run_driver(self, *extra):
        return subprocess.run([sys.executable, str(ROOT / "tools/run_experiment.py"),
                               "--config", str(self.config), "--data-root", str(self.data),
                               "--build-dir", str(self.root), "--output-dir", str(self.root / "result"),
                               *extra], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_complete_run_keeps_commands_logs_and_measured_mode(self):
        result = self.run_driver()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / "result"
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "complete")
        self.assertTrue(all(job["returncode"] == 0 for job in metadata["jobs"]))
        with (output / "summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["recall_mode"] for row in rows}, {"distance-aware"})
        self.assertTrue((output / "commands.sh").is_file())
        self.assertIn(str(self.data / "queries"), metadata["jobs"][0]["command"])

    def test_failure_retains_completed_rows_and_child_status(self):
        self.binary("rabitq_cpu", code=7)
        result = self.run_driver()
        self.assertNotEqual(result.returncode, 0)
        output = self.root / "result"
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["jobs"][1]["returncode"], 7)
        with (output / "summary.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)
        self.assertTrue((output / "cpu.log").is_file())

    def test_incorrect_recall_label_is_rejected(self):
        self.binary("coraanns_search", mode="id")
        result = self.run_driver()
        self.assertNotEqual(result.returncode, 0)
        metadata = json.loads((self.root / "result/metadata.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertIn("recall mode", metadata["error"])

    def test_incomplete_probe_table_is_rejected(self):
        self.binary("coraanns_search", incomplete=True)
        self.assertNotEqual(self.run_driver().returncode, 0)
        self.assertFalse((self.root / "result/summary.csv").exists())

    def test_id_mode_does_not_require_base_and_is_labelled(self):
        (self.data / "base").unlink()
        result = self.run_driver("--recall-mode", "id")
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / "result/summary.csv").open() as stream:
            self.assertEqual({row["recall_mode"] for row in csv.DictReader(stream)}, {"id"})

    def test_missing_base_stops_before_creating_results(self):
        (self.data / "base").unlink()
        self.assertNotEqual(self.run_driver().returncode, 0)
        self.assertFalse((self.root / "result").exists())

    def test_existing_output_is_not_overwritten(self):
        output = self.root / "result"
        output.mkdir()
        (output / "marker").write_text("existing result")
        self.assertNotEqual(self.run_driver().returncode, 0)
        self.assertEqual(list(output.iterdir()), [output / "marker"])

    def test_dry_run_needs_no_inputs_and_creates_no_output(self):
        for path in self.data.iterdir():
            path.unlink()
        result = self.run_driver("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "result").exists())


if __name__ == "__main__":
    unittest.main()
