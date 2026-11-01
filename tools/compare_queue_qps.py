#!/usr/bin/env python3
"""Compare repeated, configuration-matched original/ring QPS measurements."""

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path


def read_log(path):
    text = path.read_text()
    config = dict(re.findall(r"^([a-z_]+):\s*(.+)$", text, re.M))
    points = {}
    for round_, probe, qps in re.findall(
        r"\[Run\] round=(\d+) nprobe=(\d+) QPS=([\d.eE+-]+)", text
    ):
        points.setdefault(int(probe), []).append((int(round_), float(qps)))
    recalls = {int(n): float(r) for n, r in re.findall(
        r"\[RecallPoint\] nprobe=(\d+) source_round=\d+ recall=([\d.eE+-]+)", text
    )}
    queues = {}
    for round_, probe, submitted, spilled, wraps in re.findall(
        r"\[QueueStats\] round=(\d+) nprobe=(\d+) submitted=(\d+) spilled=(\d+) wraps=(\d+)", text
    ):
        queues.setdefault(int(probe), []).append(dict(
            round=int(round_), submitted=int(submitted), spilled=int(spilled), wraps=int(wraps)))
    if not points or set(points) != set(recalls):
        raise ValueError(f"{path}: incomplete QPS/recall measurements")
    for probe, runs in points.items():
        if sorted(r for r, _ in runs) != list(range(1, int(config["rounds"]) + 1)):
            raise ValueError(f"{path}: missing/duplicate rounds at nprobe={probe}")
        points[probe] = [q for _, q in sorted(runs)]
        if not all(math.isfinite(q) and q > 0 for q in points[probe]):
            raise ValueError(f"{path}: invalid QPS at nprobe={probe}")
    if not all(math.isfinite(r) and 0 <= r <= 1 for r in recalls.values()):
        raise ValueError(f"{path}: invalid recall")
    return dict(path=str(path), config=config, points=points, recalls=recalls,
                queues=queues, recall_mode=("distance-aware" if
                    "[Recall] mode=distance-aware" in text else "id"))


def mean_samples(groups, rng):
    chosen = [rng.choice(groups) for _ in groups]
    return statistics.mean(rng.choice(group) for group in chosen for _ in group)


def compare(before, after, *, ring_after=True, require_wrap=False):
    keys = ("topk", "batch_q", "query_chunk", "cpu_threads", "vanguard_size",
            "cpu_secondary_filter", "cross_query_reuse", "rounds", "warmup_rounds",
            "profile", "candidate_stats", "gpu_filter_timing", "nprobes")
    reference = before[0]
    for log in before + after:
        for key in keys:
            if key not in log["config"]:
                raise ValueError(f"{log['path']}: missing configuration: {key}")
            if log["config"].get(key) != reference["config"].get(key):
                raise ValueError(f"{log['path']}: configuration mismatch: {key}")
        if log["recall_mode"] != reference["recall_mode"]:
            raise ValueError(f"{log['path']}: mismatched recall mode")
        if set(log["points"]) != set(reference["points"]):
            raise ValueError(f"{log['path']}: mismatched nprobe list")
    if len(before) < 2 or len(after) < 2:
        raise ValueError("Acceptance requires at least two independent runs per version")
    if int(reference["config"]["rounds"]) < 5:
        raise ValueError("Acceptance requires at least five measured rounds per run")
    if ring_after:
        for log in after:
            for key in ("queue_capacity", "queue_spill_capacity"):
                if key not in log["config"] or log["config"][key] != after[0]["config"].get(key):
                    raise ValueError(f"{log['path']}: missing/mismatched {key}")
            for probe in log["points"]:
                rounds = sorted(q["round"] for q in log["queues"].get(probe, []))
                if rounds != list(range(1, int(log["config"]["rounds"]) + 1)):
                    raise ValueError(f"{log['path']}: missing/duplicate queue rounds at nprobe={probe}")
    rng = random.Random(20261031)
    rows = []
    for probe in reference["points"]:
        a = [log["points"][probe] for log in before]
        b = [log["points"][probe] for log in after]
        original = statistics.mean(v for g in a for v in g)
        ring = statistics.mean(v for g in b for v in g)
        ratios = sorted(mean_samples(b, rng) / mean_samples(a, rng)
                        for _ in range(10000))
        low, high = ratios[250], ratios[9749]
        baseline_means = [statistics.mean(g) for g in a]
        baseline_span = (max(baseline_means) - min(baseline_means)) / original
        status = "PASS" if ring >= original else (
            "WITHIN_NOISE" if low <= 1 <= high else "FAIL")
        if baseline_span > .03:
            status = "UNSTABLE_BASELINE"
        queue = [q for log in after for q in log["queues"].get(probe, [])]
        if ring_after and len(queue) != sum(len(g) for g in b):
            raise ValueError(f"nprobe={probe}: missing ring queue statistics")
        if any(q["spilled"] != 0 for q in queue):
            status = "SPILL_DETECTED"
        if require_wrap and (not queue or any(q["wraps"] == 0 for q in queue)):
            status = "MISSING_WRAP_COVERAGE"
        recall_before = statistics.mean(log["recalls"][probe] for log in before)
        recall_after = statistics.mean(log["recalls"][probe] for log in after)
        if recall_after < recall_before - .0001:
            status = "RECALL_REGRESSION"
        rows.append(dict(nprobe=probe, before_qps=original, after_qps=ring,
                         change_percent=100 * (ring / original - 1),
                         ratio_ci95=[low, high], baseline_span_percent=100*baseline_span,
                         recall_before=recall_before, recall_after=recall_after,
                         recall_mode=reference["recall_mode"],
                         min_wraps=min((q["wraps"] for q in queue), default=0),
                         max_spilled=max((q["spilled"] for q in queue), default=0), status=status))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, nargs="+", required=True)
    parser.add_argument("--after", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linear-after", action="store_true",
                        help="compare two linear versions before migrating the queue")
    parser.add_argument("--require-wrap", action="store_true",
                        help="require actual slot reuse in every measured round")
    args = parser.parse_args()
    if args.linear_after and args.require_wrap:
        parser.error("--require-wrap requires a ring version")
    rows = compare([read_log(p) for p in args.before], [read_log(p) for p in args.after],
                   ring_after=not args.linear_after, require_wrap=args.require_wrap)
    args.output.write_text(json.dumps(dict(before=[str(p) for p in args.before],
        after=[str(p) for p in args.after], points=rows), indent=2) + "\n")
    print("nprobe\tbefore_QPS\tafter_QPS\tchange\t95% ratio CI\twraps\tstatus")
    for row in rows:
        print(f"{row['nprobe']}\t{row['before_qps']:.1f}\t{row['after_qps']:.1f}\t"
              f"{row['change_percent']:+.2f}%\t{row['ratio_ci95']}\t"
              f"{row['min_wraps']}\t{row['status']}")
    return 0 if all(r["status"] in ("PASS", "WITHIN_NOISE") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
