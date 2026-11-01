#!/usr/bin/env python3
"""Check imported files and recorded revisions against their SHA-256 snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from cpp_tokens import tokens


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/import_manifest.json").read_text())
    errors = []
    revised = 0
    compared = 0
    for entry in manifest["files"]:
        revision = entry.get("revision")
        revised += bool(revision)
        path = root / entry["destination"]
        expected_hash = revision["sha256"] if revision else entry["imported_sha256"]
        if not path.is_file() or digest(path) != expected_hash:
            errors.append(f"snapshot differs: {entry['destination']}")
        if args.source_root:
            path = args.source_root / entry["source"]
            accepted_hashes = {entry["source_sha256"]}
            if revision:
                accepted_hashes.add(revision["reference_source_sha256"])
            if not path.is_file() or digest(path) not in accepted_hashes:
                errors.append(f"source differs: {entry['source']}")
            elif entry.get("compare_cpp_tokens") and not revision:
                expected = path.read_text()
                if "source_character_range" in entry:
                    begin, end = entry["source_character_range"]
                    expected = expected[begin:end]
                for old, new in manifest["include_replacements"].items():
                    expected = expected.replace(old, new)
                expected = entry.get("added_prefix", "") + expected
                if tokens(expected) != tokens((root / entry["destination"]).read_text()):
                    errors.append(f"C++ tokens differ: {entry['destination']}")
                compared += 1
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    message = f"Verified {len(manifest['files']) - revised} unchanged imports and {revised} recorded revisions"
    if args.source_root:
        message += f", source snapshots, and {compared} unchanged C++/CUDA token sequences"
    print(message + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
