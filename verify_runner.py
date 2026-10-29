#!/usr/bin/env python3
"""Verify that a self-hosted runner is suitable for private model experiments."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path


def available_accelerator() -> str | None:
    try:
        import torch
    except ImportError:
        print(
            "PyTorch is missing from the active Python environment; "
            "run uv sync --python 3.14 --extra dev --extra model --locked "
            "before neural training",
            file=sys.stderr,
        )
        return None
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps:Apple Metal"
    print(
        f"runner={platform.system()} {platform.machine()}, "
        f"python={sys.version.split()[0]}, torch={torch.__version__}",
        file=sys.stderr,
    )
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="Allow CPU-only execution for acoustic candidates that do not use PyTorch",
    )
    arguments = parser.parse_args()

    manifest = Path(arguments.manifest).expanduser().resolve()
    if not manifest.is_file():
        raise SystemExit(f"dataset manifest does not exist: {manifest}")
    if manifest.name == "sample_manifest.csv":
        raise SystemExit("scheduled model research refuses the synthetic sample manifest")
    accelerator = available_accelerator()
    if accelerator is None:
        if not arguments.allow_cpu:
            raise SystemExit(
                "no CUDA or Apple Metal accelerator is available; confirm the runner uses "
                "a GPU-capable PyTorch build and the Mac Metal device is accessible"
            )
        accelerator = "cpu (acoustic candidate)"
    print(f"runner={platform.system()} {platform.machine()}")
    print(f"accelerator={accelerator}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
