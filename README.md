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
