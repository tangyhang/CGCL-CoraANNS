#!/usr/bin/env python3
"""Run a serial A/B/B/A queue comparison and save reproducible evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from compare_queue_qps import compare, read_log
from run_experiment import nonnegative, positive, ring_capacity

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--dataset", choices=["deep100m", "wiki88m", "laion100m"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=positive, default=72)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--queue-capacity", type=ring_capacity, default=1 << 24)
    parser.add_argument("--queue-spill-capacity", type=nonnegative, default=1 << 24)
    parser.add_argument("--recall-mode", choices=["distance-aware", "id"], default="distance-aware")
    parser.add_argument("--linear-after", action="store_true", help="compare two linear implementations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = json.loads((ROOT / "configs/queue_benchmarks.json").read_text())[args.dataset]
    files = {key: (args.data_root / name).resolve() for key, name in settings["files"].items()}
    binaries = {"a": args.before.resolve(), "b": args.after.resolve()}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS=str(args.threads),
               OMP_DYNAMIC="FALSE", OMP_PROC_BIND="close", OMP_PLACES="cores")
    env_keys = ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES"]
    jobs = []
    for label, version in [("a1", "a"), ("b1", "b"), ("b2", "b"), ("a2", "a")]:
        command = ["numactl", "--interleave=all", str(binaries[version]),
                   str(files["index"]), str(files["queries"]), str(files["groundtruth"]),
                   "--topk", "100", "--nprobes", settings["nprobes"],
                   "--rounds", "5", "--warmup", "1", "--batch-q", "4",
                   "--query-chunk", str(settings["query_chunk"]), "--threads", str(args.threads),
                   "--vanguard-size", "5", "--cpu-secondary-filter", "true",
                   "--cross-query-reuse", "true", "--profile", "false", "--candidate-stats", "false",
                   "--gpu-filter-timing", "false", "--recall-once", "true"]
        if args.recall_mode == "distance-aware":
            command += ["--recall-base", str(files["base"])]
        if version == "b" and not args.linear_after:
            command += ["--queue-capacity", str(args.queue_capacity),
                        "--queue-spill-capacity", str(args.queue_spill_capacity)]
        jobs.append(dict(label=label, command=command))
        print(label, shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    required = ["index", "queries", "groundtruth"]
    if args.recall_mode == "distance-aware":
        required.append("base")
    for path in [*binaries.values(), *(files[key] for key in required)]:
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"missing or empty file: {path}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="running", dataset=args.dataset, settings=settings,
                    recall_mode=args.recall_mode, jobs=jobs,
                    environment={key: env[key] for key in env_keys},
                    binaries={key: dict(path=str(path), sha256=sha256(path)) for key, path in binaries.items()},
                    inputs={key: dict(path=str(files[key]), bytes=files[key].stat().st_size,
                                      mtime_ns=files[key].stat().st_mtime_ns) for key in required})
    write_json(output / "metadata.json", metadata)
    (output / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexport " +
        shlex.join([f"{key}={env[key]}" for key in env_keys]) + "\n" +
        "\n".join(shlex.join(job["command"]) for job in jobs) + "\n")
    try:
        for job in jobs:
            started = time.monotonic()
            print("START", job["label"], flush=True)
            with (output / f"{job['label']}.log").open("w") as stream:
                result = subprocess.run(job["command"], env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=1800)
            job.update(returncode=result.returncode, seconds=time.monotonic() - started)
            write_json(output / "metadata.json", metadata)
            if result.returncode:
                raise RuntimeError(f"{job['label']} exited with {result.returncode}")
            print("DONE", job["label"], round(job["seconds"], 2), "seconds", flush=True)
        rows = compare([read_log(output / f"{label}.log") for label in ["a1", "a2"]],
                       [read_log(output / f"{label}.log") for label in ["b1", "b2"]],
                       ring_after=not args.linear_after,
                       require_wrap=not args.linear_after and settings["require_wrap"])
        accepted = all(row["status"] in {"PASS", "WITHIN_NOISE"} for row in rows)
        write_json(output / "comparison.json", dict(points=rows, accepted=accepted))
        metadata.update(status="complete", accepted=accepted)
        for row in rows:
            print(row["nprobe"], f"{row['change_percent']:+.2f}%", row["status"], flush=True)
        return 0 if accepted else 1
    except BaseException as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "metadata.json", metadata)


if __name__ == "__main__":
    raise SystemExit(main())
