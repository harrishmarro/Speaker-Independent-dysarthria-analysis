# Security policy

## Reporting

Do not open a public issue for a vulnerability involving participant data, credentials,
model artifacts, or code execution. Use GitHub's private vulnerability reporting feature.

## Supported code

Security fixes apply to the current `main` branch. Research commits and model artifacts are
not releases unless they have passed the documented security, privacy, reproducibility, and
human-review gates.

## Protected boundaries

- Raw clinical audio and identifying metadata must remain outside Git.
- Autonomous candidate code may write only to `artifacts/` through locked I/O helpers.
- Candidate code cannot use network, shell, subprocess, dynamic execution, unsafe
  deserialization, or direct arbitrary file access.
- Model sources, revisions, licenses, and cryptographic hashes must be recorded before use.
- Held-out test labels remain unavailable during autonomous iteration.

No model produced by this repository is approved for clinical or diagnostic use.
