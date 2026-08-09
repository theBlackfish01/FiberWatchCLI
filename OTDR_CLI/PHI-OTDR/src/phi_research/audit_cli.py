"""Command-line entry point for Phi-OTDR inventory and split creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_contract import audit_dataset, create_split_from_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    summary = audit_dataset(args.data_root, args.output_dir)
    manifest = create_split_from_audit(args.output_dir, args.manifest, seed=args.seed)
    print(json.dumps({"audit": summary, "split_summary": manifest["summary"], "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
