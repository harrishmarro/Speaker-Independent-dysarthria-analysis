# Speaker-Independent Dysarthria Speech Analytics

An explainable research prototype for dysarthria detection and severity analysis using
WavLM representations, a Mixture-of-Experts classifier, and ASR-derived intelligibility
signals.

**Author and project owner:** Harish P. S.

> **Status:** research implementation in progress. The architecture and evaluation
> protocol are defined, but model performance has not yet been independently reproduced.
> No clinical-use claim is made.

## Why this project exists

Dysarthria can affect articulation, phonation, timing, and speech intelligibility in
different ways across speakers. This project investigates whether a dynamically routed
Mixture-of-Experts layer can model that variation while a strict speaker-wise split
measures generalization to unseen speakers. A separate ASR analysis provides interpretable
word-error and confidence signals.

## Proposed system

```text
16 kHz audio
    │
    ▼
WavLM encoder ──► frame representations ──► Mixture of Experts
                                                   │
                                                   ▼
                                      dysarthria probability
                                                   │
reference transcript ──► ASR ──► WER ──────────────┤
                                                   ▼
                                      severity analysis
```

The acoustic classifier and ASR analysis are evaluated separately before any fused
severity score is reported. This avoids presenting a heuristic score as a clinically
validated severity measure.

## Repository contents

- `program.md` — human-maintained instructions for the autonomous research loop
- `prepare.py` — locked data-validation and speaker-split entry point
- `train.py` — the only model experiment file an autonomous researcher may modify
- `evaluate.py` — locked evaluation entry point
- `research.py` — controlled experiment runner and append-only registry writer
- `security.py` — static policy gate applied before candidate code executes
- `src/dysarthria/manifest.py` — validates dataset manifests and creates deterministic,
  speaker-disjoint splits
- `src/dysarthria/model.py` — WavLM and Mixture-of-Experts architecture
- `src/dysarthria/metrics.py` — transparent classification and word-error calculations
- `src/dysarthria/severity.py` — configurable experimental fusion rule
- `configs/base.yaml` — declared experiment configuration
- `data/sample_manifest.csv` — synthetic manifest showing the required schema
- `paper/research_proposal.md` — corrected sole-author research proposal
- `docs/research_integrity.md` — claim, privacy, and evaluation rules
- `tests/` — tests for leakage prevention and metric correctness

## Dataset contract

The real audio is deliberately excluded. Each local row must contain:

| Column | Meaning |
| --- | --- |
| `audio_path` | Local path to a 16 kHz recording |
| `speaker_id` | Stable pseudonymous speaker identifier |
| `label` | `control` or `dysarthria` |
| `severity` | Optional source severity label |
| `transcript` | Optional reference transcript for WER |

Do not commit clinical recordings, names, raw participant identifiers, consent records,
or other protected data.

## Local validation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
dysarthria-data validate data/sample_manifest.csv
dysarthria-data split data/sample_manifest.csv --output data/sample_splits.csv
```

Model training dependencies are optional because WavLM downloads are large:

```bash
python -m pip install -e ".[model]"
```

## Secure autoresearch loop

The design adapts the small, controlled experiment loop from `karpathy/autoresearch` to
sensitive speech research. The research agent may change `train.py` and declared research
configuration only. Dataset splits, evaluation, privacy rules, security policy, and test
labels are locked.

```bash
python research.py --mode smoke
```

The scheduled GitHub workflow checks for an eligible cycle every ten minutes on a self-hosted runner
labelled `dysarthria-gpu`. It requires a private licensed dataset manifest and either CUDA or
Apple Metal acceleration. See `docs/local-gpu-runner.md`. The synthetic manifest is accepted
only for manual smoke validation and is never promoted to a research finding.

The schedule requests a cycle every ten minutes, and each completed candidate queues the next
candidate through its search-state commit. The current plan compares speaker-level nearest-
neighbor classifiers with 1, 3, and 5 neighbors. The controller tries one variant from a
finite, reviewed set in `autoresearch/decide.py`, compares validation balanced accuracy
with the accepted baseline, and records each outcome. Equal or worse variants restore the
baseline configuration on `main`. A better variant opens a pull request for review before
it can become the new baseline. The loop stops when the set is exhausted. This is bounded
automated search; the runner does not yet write new Python model code.

`experiments/registry.tsv` records each run's source commit, model-code commit, candidate
configuration hash, and configuration values. Older rows have no verifiable run commit;
their model-code commit is backfilled from the exact `train.py` content hash.
`experiments/Model_used.tsv` preserves accepted baselines when the search moves to a new
approach. Only the best retained approach from a previous search remains available for
comparison; rejected variants are kept as result rows, not active candidates.

## Evaluation rules

1. Split by speaker before feature extraction or model fitting.
2. Report speaker counts and class balance for every split.
3. Tune only on the validation speakers; evaluate test speakers once.
4. Report accuracy, balanced accuracy, precision, recall, F1, ROC AUC, and confusion
   counts with uncertainty across multiple seeds.
5. Compare WavLM + MoE with meaningful baselines and ablations.
6. Keep proposed, implemented, and experimentally validated work clearly separated.

## Responsible-use statement

This repository is for research and portfolio demonstration. It is not a diagnostic
device and must not guide clinical decisions. Any study involving participant speech
must follow the dataset license, consent terms, institutional requirements, and applicable
privacy law.
