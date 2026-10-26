"""Command-line tools for manifest validation and splitting."""

from __future__ import annotations

import argparse

from dysarthria.manifest import assign_speaker_splits, read_manifest, write_manifest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="dysarthria-data")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a dataset manifest")
    validate.add_argument("manifest")
    split = commands.add_parser("split", help="create deterministic speaker-wise splits")
    split.add_argument("manifest")
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=42)
    return root


def main() -> None:
    arguments = parser().parse_args()
    rows = read_manifest(arguments.manifest)
    if arguments.command == "validate":
        print(f"valid manifest: {len(rows)} recordings")
        return
    split_rows = assign_speaker_splits(rows, seed=arguments.seed)
    write_manifest(split_rows, arguments.output)
    print(f"wrote {len(split_rows)} recordings to {arguments.output}")


if __name__ == "__main__":
    main()
