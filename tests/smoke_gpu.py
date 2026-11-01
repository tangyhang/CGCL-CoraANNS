"""Build CPU/GPU indexes, load the packed adapter and check both refinement modes."""
import argparse
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile

from smoke_cpu import write_bin, write_vecs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexers", nargs="+", required=True)
    parser.add_argument("--search", required=True)
    args = parser.parse_args()
    rng = random.Random(42)
    dim, count, nlist = 96, 1024, 4
    centers = [[rng.uniform(-4, 4) for _ in range(dim)] for _ in range(nlist)]
    base = [[centers[i % nlist][d] + rng.gauss(0, 0.1) for d in range(dim)]
            for i in range(count)]
    selected = [3, 17, 205, 511, 1001]
    env = dict(os.environ, OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE")
    with tempfile.TemporaryDirectory(prefix="coraanns-gpu-smoke-") as name:
        root = Path(name)
        write_bin(root / "base.fbin", base, "f")
        write_bin(root / "queries.fbin", [base[i] for i in selected], "f")
        write_bin(root / "gt.ibin", [[i] for i in selected], "I")
        write_vecs(root / "centers.fvecs", centers, "f")
        write_vecs(root / "assignments.ivecs", [[i % nlist] for i in range(count)], "I")
        for i, indexer in enumerate(args.indexers):
            index = root / f"ivf-{i}.index"
            subprocess.run([indexer, str(root / "base.fbin"), str(root / "centers.fvecs"),
                str(root / "assignments.ivecs"), "8", str(index), "l2", "false"], check=True, env=env)
            for exact in [False, True]:
                command = [args.search, str(index), str(root / "queries.fbin"), str(root / "gt.ibin"),
                    "--topk", "1", "--nprobes", "4", "--threads", "2", "--warmup", "0",
                    "--rounds", "1", "--query-chunk", "2", "--queue-capacity", "256",
                    "--queue-spill-capacity", "8192", "--recall-base", str(root / "base.fbin")]
                if exact:
                    command += ["--exact-base", str(root / "base.fbin")]
                result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, env=env)
                print(result.stdout)
                rows = re.findall(r"^4\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$", result.stdout, re.M)
                if not rows or float(rows[-1][0]) <= 0 or float(rows[-1][1]) != 1.0:
                    raise RuntimeError("GPU pipeline did not recover known nearest vectors")


if __name__ == "__main__":
    main()
