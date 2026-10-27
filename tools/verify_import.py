#!/usr/bin/env python3
"""Check imported file hashes, optionally against the original working tree."""
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
    for entry in manifest["files"]:
        path = root / entry["destination"]
        if not path.is_file() or digest(path) != entry["imported_sha256"]:
            errors.append(f"import differs: {entry['destination']}")
        if args.source_root:
            path = args.source_root / entry["source"]
            if not path.is_file() or digest(path) != entry["source_sha256"]:
                errors.append(f"source differs: {entry['source']}")
            elif entry.get("compare_cpp_tokens"):
                expected = path.read_text()
                if "source_character_range" in entry:
                    begin, end = entry["source_character_range"]
                    expected = expected[begin:end]
                for old, new in manifest["include_replacements"].items():
                    expected = expected.replace(old, new)
                expected = entry.get("added_prefix", "") + expected
                if tokens(expected) != tokens((root / entry["destination"]).read_text()):
                    errors.append(f"C++ tokens differ: {entry['destination']}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    message = f"Verified {len(manifest['files'])} imported files"
    if args.source_root:
        compared = sum(bool(entry.get("compare_cpp_tokens")) for entry in manifest["files"])
        message += f", their original sources, and {compared} unchanged C++/CUDA token sequences"
    print(message + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
