"""Create or verify frozen cross-acquisition Phi-OTDR manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .era_contract import (
    ERAS,
    create_acquisition_manifest,
    verify_acquisition_manifest,
    verify_protocol_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--audit-dir", type=Path, required=True)
    create.add_argument("--legacy-manifest", type=Path)
    create.add_argument("--source-era", choices=ERAS, required=True)
    create.add_argument("--target-era", choices=ERAS, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--seed", type=int, default=20260805)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--audit", type=Path)
    verify_protocol = subparsers.add_parser("verify-protocol")
    verify_protocol.add_argument("--protocol", type=Path, required=True)
    verify_protocol.add_argument("--hash", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "create":
        manifest = create_acquisition_manifest(
            args.audit_dir,
            args.output,
            source_era=args.source_era,
            target_era=args.target_era,
            seed=args.seed,
            legacy_manifest_path=args.legacy_manifest,
        )
        expected = None
    elif args.command == "verify":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        expected = None
        if args.audit:
            expected = str(json.loads(args.audit.read_text(encoding="utf-8"))["dataset_fingerprint_sha256"])
    else:
        print(json.dumps(verify_protocol_hash(args.protocol, args.hash), indent=2))
        return
    print(
        json.dumps(
            verify_acquisition_manifest(manifest, expected_dataset_fingerprint=expected), indent=2
        )
    )


if __name__ == "__main__":
    main()
