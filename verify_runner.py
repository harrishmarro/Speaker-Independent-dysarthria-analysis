#!/usr/bin/env python3
"""Verify that a self-hosted runner is suitable for private model experiments."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path


def available_accelerator() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps:Apple Metal"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    arguments = parser.parse_args()

    manifest = Path(arguments.manifest).expanduser().resolve()
    if not manifest.is_file():
        raise SystemExit(f"dataset manifest does not exist: {manifest}")
    if manifest.name == "sample_manifest.csv":
        raise SystemExit("scheduled model research refuses the synthetic sample manifest")
    accelerator = available_accelerator()
    if accelerator is None:
        raise SystemExit("no CUDA or Apple Metal accelerator is available")
    print(f"runner={platform.system()} {platform.machine()}")
    print(f"accelerator={accelerator}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
