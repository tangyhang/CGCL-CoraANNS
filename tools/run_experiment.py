#!/usr/bin/env python3
"""Run the preserved search executables with portable dataset configurations."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--experiment", choices=["overall", "ablation", "vanguard", "threads", "topk"], default="overall")
    parser.add_argument("--output-dir", type=Path, help="new directory; existing results are never overwritten")
    parser.add_argument("--threads", type=positive, help="override worker count except in the threads sweep")
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--numa-interleave", action="store_true", help="prefix each command with numactl --interleave=all")
    parser.add_argument("--recall-mode", choices=["distance-aware", "id"], default="distance-aware")
    parser.add_argument("--dry-run", action="store_true", help="print commands without loading data or creating results")
    return parser


def make_jobs(config, args):
    settings = config["search"]
    spec = config["experiments"][args.experiment]
    files = {key: str((args.data_root / value).resolve())
             for key, value in config["files"].items()}
    if "groundtruth" in spec:
        files["groundtruth"] = str((args.data_root / spec["groundtruth"]).resolve())
    threads = args.threads or settings["threads"]
    prefix = ["numactl", "--interleave=all"] if args.numa_interleave else []

    def job(name, binary="coraanns_search", *, topk=None, nprobes=None, vanguard=None, secondary=None, workers=None):
        k = settings["topk"] if topk is None else topk
        probes = list(spec["nprobes"] if nprobes is None else nprobes)
        if not probes or len(set(probes)) != len(probes) or any(
                type(p) is not int or p < 1 or p > config["nlist"] for p in probes):
            raise ValueError("nprobes must be unique integers in [1, nlist]")
        if type(k) is not int or k < 1 or k > 1024:
            raise ValueError("topk must be in [1, 1024]")
        if binary in {"coraanns_search", "rabitq_gpu"} and k > 512:
            binary += "_highk"
        command = prefix + [str((args.build_dir / "bin" / binary).resolve()),
                            files["index"], files["queries"], files["groundtruth"],
                            "--topk", str(k), "--nprobes", ",".join(map(str, probes)),
                            "--rounds", str(settings["rounds"]),
                            "--warmup", str(settings["warmup"]), "--profile", "false"]
        if binary.startswith("coraanns_search"):
            vg = settings["vanguard_size"] if vanguard is None else vanguard
            cf = settings["cpu_secondary_filter"] if secondary is None else secondary
            command += ["--threads", str(threads if workers is None else workers),
                        "--batch-q", str(settings["batch_q"]),
                        "--query-chunk", str(settings["query_chunk"]),
                        "--vanguard-size", str(vg),
                        "--cpu-secondary-filter", str(cf).lower(),
                        "--cross-query-reuse", str(settings["cross_query_reuse"]).lower()]
        elif binary.startswith("rabitq_gpu"):
            command += ["--query-chunk", str(settings["gpu_query_chunk"]),
                        "--index-residency", "host-excode", "--fine-kernel", "tiled"]
        elif binary == "rabitq_cpu":
            command += ["--threads", str(threads), "--hacc", "true"]
        if args.recall_mode == "distance-aware":
            command += ["--recall-base", files["base"]]
        return {"name": name, "command": command, "binary": binary,
                "nprobes": probes, "topk": k}

    if args.experiment == "overall":
        jobs = [job("coraanns"), job("cpu", "rabitq_cpu"), job("gpu_host_excode", "rabitq_gpu")]
    elif args.experiment == "ablation":
        jobs = [job("baseline", "rabitq_gpu"),
                job("technique1", vanguard=0, secondary=False),
                job("technique1_2", vanguard=5, secondary=False),
                job("technique1_3", vanguard=0, secondary=True),
                job("all", vanguard=5, secondary=True)]
    elif args.experiment == "vanguard":
        jobs = [job(f"vanguard_{size}", vanguard=size) for size in spec["sizes"]]
    elif args.experiment == "threads":
        jobs = [job(f"threads_{workers}", workers=workers, vanguard=spec["vanguard_size"])
                for workers in spec["threads"]]
    else:
        jobs = [job(f"top{item['topk']}", topk=item["topk"], nprobes=item["nprobes"])
                for item in spec["jobs"]]
    if len({j["name"] for j in jobs}) != len(jobs):
        raise ValueError("job names must be unique")
    return files, jobs


def parse_summary(text, expected):
    """Use the executable's final mean-QPS table, preserving its timer semantics."""
    marker = "nprobe\tQPS\trecall"
    if marker not in text:
        raise ValueError("search did not emit the final nprobe/QPS/recall table")
    rows = []
    for line in text.rsplit(marker, 1)[1].splitlines():
        fields = line.split()
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        nprobe, qps, recall = int(fields[0]), float(fields[1]), float(fields[2])
        if not math.isfinite(qps) or qps <= 0 or not math.isfinite(recall) or not 0 <= recall <= 1:
            raise ValueError("invalid QPS or recall in final table")
        rows.append({"nprobe": nprobe, "qps": qps, "recall": recall})
    if [row["nprobe"] for row in rows] != expected:
        raise ValueError("final table does not match the requested nprobe list")
    return rows


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_summary(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "experiment", "job", "topk",
                                                   "nprobe", "qps", "recall", "recall_mode", "log"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    files, jobs = make_jobs(config, args)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_DYNAMIC="FALSE",
               OMP_PROC_BIND="close", OMP_PLACES="cores",
               OMP_NUM_THREADS=str(args.threads or config["search"]["threads"]))
    env_keys = ["CUDA_VISIBLE_DEVICES", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES", "OMP_NUM_THREADS"]
    print(shlex.join([f"{key}={env[key]}" for key in env_keys]), flush=True)
    for job in jobs:
        print(shlex.join(job["command"]), flush=True)
    if args.dry_run:
        return 0

    if args.numa_interleave and not shutil.which("numactl"):
        parser.error("--numa-interleave requires numactl on PATH")
    required = ["index", "queries", "groundtruth"]
    if args.recall_mode == "distance-aware":
        required.append("base")
    for key in required:
        path = Path(files[key])
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"missing or empty {key} file: {path}")
    for job in jobs:
        binary = args.build_dir / "bin" / job["binary"]
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f"build target {job['binary']} first: {binary}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or ROOT / "results" / config["dataset"] / f"{args.experiment}_{stamp}_{uuid.uuid4().hex[:8]}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "started_utc": stamp, "config": config,
                "files": {key: {"path": files[key], "bytes": Path(files[key]).stat().st_size,
                                 "mtime_ns": Path(files[key]).stat().st_mtime_ns} for key in required},
                "environment": {key: env[key] for key in env_keys},
                "platform": platform.platform(), "python": sys.version,
                "recall_mode": args.recall_mode, "experiment": args.experiment,
                "import_manifest_sha256": hashlib.sha256((ROOT / "docs/import_manifest.json").read_bytes()).hexdigest(),
                "jobs": jobs}
    write_json(output / "metadata.json", metadata)
    (output / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" +
        "export " + shlex.join([f"{key}={env[key]}" for key in env_keys]) + "\n" +
        "\n".join(shlex.join(j["command"]) for j in jobs) + "\n")
    rows = []
    try:
        for job in jobs:
            log_path = output / f"{job['name']}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(job["command"], env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace")
                try:
                    for line in process.stdout:
                        log.write(line)
                        print(line, end="", flush=True)
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                finally:
                    process.stdout.close()
            job["returncode"] = code
            if code:
                raise RuntimeError(f"{job['name']} exited with {code}; see {log_path}")
            text = log_path.read_text()
            if f"[Recall] mode={args.recall_mode}" not in text:
                raise RuntimeError(f"recall mode was not confirmed in {log_path}")
            for row in parse_summary(text, job["nprobes"]):
                rows.append(dict(dataset=config["dataset"], experiment=args.experiment,
                                 job=job["name"], topk=job["topk"], recall_mode=args.recall_mode,
                                 log=log_path.name, **row))
            write_summary(output / "summary.csv", rows)
            write_json(output / "metadata.json", metadata)
        metadata["status"] = "complete"
    except BaseException as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "metadata.json", metadata)
        print(f"Results: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        sys.exit(f"error: {error}")
