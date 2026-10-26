# Secure dysarthria autoresearch program

You are conducting controlled research on speaker-independent dysarthria analysis.

## Objective

Improve validation-speaker performance and reliability under the locked evaluation protocol.
Prefer simple, testable changes. Preserve negative and inconclusive results.

## Files you may modify

- `train.py`
- `configs/research/`
- the current directory under `experiments/autonomous/`

All other code is infrastructure or research evidence. Do not modify locked data preparation,
speaker splits, evaluation, security controls, tests, or prior experiment records.

## Prohibited actions

- Do not access held-out test labels during iteration.
- Do not install dependencies or download models during an experiment.
- Do not use subprocesses, shell commands, dynamic code execution, network calls, or unsafe
  checkpoint deserialization in candidate code.
- Do not read environment secrets or files outside the repository and declared data roots.
- Do not commit audio, participant metadata, embeddings, checkpoints, or predictions.
- Do not describe a smoke test, proposal, or single run as a validated finding.

## Experiment loop

1. Read the registry, negative results, and current candidate implementation.
2. State one falsifiable hypothesis and its expected tradeoff.
3. Change only the allowed candidate surface.
4. Run `python security.py train.py`.
5. Run unit, leakage, privacy, and reproducibility tests.
6. Commit the candidate change.
7. Run `python research.py --mode experiment` under the fixed resource budget.
8. Record the result even when it crashes, regresses, or is inconclusive.
9. Retain a candidate only when all hard gates pass and the locked primary metric improves.
10. Send retained candidates through repeated-seed confirmation before requesting a held-out
    test evaluation.

## Scientific standard

An experiment result is evidence only for its declared configuration, dataset snapshot,
speaker split, seed, and compute environment. A finding requires confirmation across the
declared seeds and a review of calibration, error cases, subgroup behavior, and security.
