---
name: speaker-independent-dysarthria-autoresearch
description: Run secure, reproducible, speaker-independent dysarthria research with a controlled WavLM and Mixture-of-Experts experiment loop on a local GPU runner.
---

# Speaker-independent dysarthria autoresearch

This skill governs the research repository:

`https://github.com/harrishmarro/Speaker-Independent-dysarthria-analysis`

The project is owned and authored by **Harish P. S.** Prior researchers remain credited
only through scholarly citations. Do not add other people as project authors unless Harish
explicitly provides written attribution and contribution details.

The repository adapts the controlled experiment-loop idea from `karpathy/autoresearch`.
Its purpose is to run repeatable research experiments, preserve negative results, and
promote findings only after independent verification. It is not a clinical device and does
not produce medical diagnoses.

## Core rules

1. Treat the supplied research paper as a proposal and source document until its results
   can be reproduced from code, data manifests, configurations, predictions, and metrics.
2. Keep raw speech, participant identifiers, consent records, and protected metadata out of
   Git and out of logs.
3. Split by speaker before feature extraction, training, tuning, or evaluation.
4. Keep held-out test speakers unavailable during autonomous iteration.
5. Never change locked evaluation, privacy, security, or split code to improve a metric.
6. Record every experiment, including crashes, regressions, inconclusive results, and
   rejected hypotheses.
7. Never describe a smoke run, proposal, baseline, or single seed as a validated finding.
8. Do not load untrusted Python pickle checkpoints or execute remote model code.
9. Do not put credentials, API keys, raw audio, embeddings, checkpoints, or private paths in
   commits, issues, pull requests, or chat.
10. Do not run the local GPU workflow for pull requests from forks or untrusted branches.

## Repository roles

The files have deliberately different trust levels.

### Locked infrastructure

These files define scientific and security boundaries. An autonomous researcher must not
modify them:

- `prepare.py` — manifest validation and speaker-disjoint splits
- `evaluate.py` — prediction validation and locked metrics
- `security.py` — candidate-code static policy gate
- `scheduler.py` — cycle timestamp utility
- `verify_runner.py` — local accelerator and private-manifest checks
- `locked/` — constrained candidate file access
- `tests/security/`, `tests/leakage/`, `tests/privacy/`
- `data/manifests/`, `data/splits/`
- `autoresearch/policy.yaml`, `autoresearch/objective.yaml`, and
  `autoresearch/resources.yaml`

### Candidate research surface

The research agent may modify only:

- `train.py`
- `configs/research/`
- `experiments/autonomous/<experiment-id>/`

Every candidate must pass security checks, linting, type checking, tests, and the locked
evaluation path before it can be retained.

### Evidence and reporting

- `experiments/registry.tsv` is append-only experiment history.
- `research/findings/validated.md` contains only confirmed findings.
- `research/findings/negative-results.md` contains confirmed failures and regressions.
- `research/findings/inconclusive.md` contains results that need more evidence.
- `manuscript/` contains publication drafts and citations.
- `reports/` contains reviewed aggregate figures, tables, model cards, and dataset cards.

## First-time setup

Clone the repository and enter it:

```bash
git clone https://github.com/harrishmarro/Speaker-Independent-dysarthria-analysis.git
cd Speaker-Independent-dysarthria-analysis
```

Install `uv`, then install the development environment:

```bash
uv sync --extra dev --locked
```

For WavLM and audio experiments, install the model extras as well:

```bash
uv sync --extra dev --extra model --locked
```

Run the local checks:

```bash
uv run ruff check .
uv run mypy src locked prepare.py evaluate.py research.py scheduler.py security.py verify_runner.py
uv run pytest
uv run bandit -q -r src locked prepare.py evaluate.py research.py scheduler.py security.py verify_runner.py
```

Run the smoke harness:

```bash
uv run python security.py train.py
uv run python research.py --mode smoke
```

The synthetic manifest is for infrastructure validation only. Its result must remain
labelled `smoke_only` and must never be used in the manuscript or resume.

## Private dataset setup

The real dataset must be licensed for the intended research use. Store it outside the Git
checkout. The manifest must contain at least:

```text
audio_path,speaker_id,label,severity,transcript
```

Requirements:

- `speaker_id` must be stable and pseudonymous.
- Each speaker must have exactly one classification label.
- Audio must be validated before use and resampled to 16 kHz by the data pipeline.
- `severity` may be absent or unknown; do not invent labels.
- `transcript` is optional and must be governed by the dataset license.
- The manifest must not contain names, email addresses, medical record IDs, or direct
  identifiers.
- Audio and private manifests must be readable only by the dedicated research account.

Set the local runner environment variable without committing it:

```text
DYSARTHRIA_MANIFEST=/absolute/private/path/dysarthria_manifest.csv
```

Validate a manifest locally:

```bash
uv run python prepare.py \
  --manifest "$DYSARTHRIA_MANIFEST" \
  --output artifacts/prepared_manifest.csv
```

Never copy the real manifest into `data/sample_manifest.csv`. The sample file is public
test data and must remain synthetic.

## Local Mac GPU runner

The scheduled research workflow uses a GitHub Actions self-hosted runner labelled
`dysarthria-gpu`.

Follow `docs/local-gpu-runner.md` and GitHub’s generated runner commands:

1. Open repository **Settings → Actions → Runners**.
2. Select **New self-hosted runner** and choose macOS plus the correct architecture.
3. Add the custom label `dysarthria-gpu` during runner configuration.
4. Install the runner as a service under a dedicated non-administrator account.
5. Put `DYSARTHRIA_MANIFEST` in the runner’s private `.env` file.
6. Keep the Mac powered, awake, online, and patched during expected research periods.

Validate the machine:

```bash
uv sync --extra dev --extra model --locked
uv run python verify_runner.py --manifest "$DYSARTHRIA_MANIFEST"
```

The check must report either `mps:Apple Metal` or a CUDA accelerator. The current scheduled
acoustic candidates run on CPU after the runner check; WavLM training is not implemented yet.

The public repository intentionally does not run model experiments on pull requests. This
prevents untrusted fork code from reaching the local machine. Keep branch protection enabled
for `main`, require passing checks, and review workflow-file changes carefully.

## Model supply chain

Before running WavLM:

1. Select a specific reviewed model commit, not a moving branch or `main` tag.
2. Record the model name, immutable revision, license, source URL, and SHA-256 artifact
   hashes in the experiment record.
3. Download or cache the model during a controlled preparation step.
4. Run experiments with network access disabled.
5. Set `trust_remote_code=False`.
6. Prefer `safetensors` artifacts and reject unsafe pickle checkpoints.
7. Confirm that the model is permitted for the dataset and intended research use.

The placeholder `REQUIRED_IMMUTABLE_COMMIT_SHA` in `configs/base.yaml` must be replaced by
an actual reviewed revision before model execution. Do not guess or silently substitute a
revision.

## Experiment protocol

Every experiment starts with one falsifiable hypothesis. Use this order:

1. Read `program.md`, prior registry entries, negative results, and relevant literature.
2. Define the hypothesis, expected benefit, and expected failure mode.
3. Change only the permitted candidate surface.
4. Run `uv run python security.py train.py`.
5. Run lint, type checks, unit tests, leakage tests, privacy tests, and reproducibility tests.
6. Commit the candidate on an experiment branch.
7. Run the controlled experiment with a fixed resource budget.
8. Save configuration, dataset hash, split hash, candidate hash, environment, predictions,
   metrics, and resource use.
9. Compare against the retained baseline on validation speakers and leave-one-training-speaker-out
   cross-validation. Refit feature scaling within every cross-validation fold.
10. Keep, reject, or mark inconclusive; record the decision and reasoning.
11. Confirm promising candidates across the declared seeds `[42, 43, 44, 45, 46]`.
12. Request human review before using held-out test speakers.

The workflow requests a run every ten minutes, subject to GitHub scheduling delays and the
self-hosted runner being online. Each result advances the next reviewed candidate automatically.
When a new model family needs approval or the finite catalog is exhausted, runs open one assigned
GitHub review issue and continue their scheduled checks without starting an unapproved experiment.
Approve the next approach by updating `autoresearch/approvals.json` only after Harish confirms the
proposal. GitHub-hosted runners perform CI and security scans; the self-hosted
`dysarthria-gpu` runner performs acoustic research.

## Metrics and promotion rules

The primary validation metric is speaker-level balanced accuracy. Tie-breakers are:

1. Macro F1
2. Dysarthria recall
3. ROC AUC
4. Calibration error
5. Runtime, memory, and model complexity

A candidate cannot be promoted if any hard gate fails:

- speaker overlap is nonzero;
- the candidate changes locked evaluation or security code;
- a security finding is unresolved;
- metrics are invalid or internally inconsistent;
- the candidate exceeds resource limits;
- calibration or subgroup performance regresses materially;
- predictions cannot be reproduced from the recorded configuration;
- protected data appears in an artifact or log.

The held-out test set is evaluated once, after repeated-seed confirmation and human review.
Do not tune thresholds, select models, or write findings using test results.

## Research questions and experiment order

The initial program should establish:

1. Dataset quality, class balance, speaker counts, and recording conditions.
2. MFCC plus a classical classifier baseline.
3. Frozen WavLM plus a linear classification head.
4. Partial WavLM fine-tuning.
5. WavLM plus Mixture-of-Experts routing.
6. Expert-count, pooling, freezing, and routing ablations.
7. ASR transcription and word-error analysis.
8. Calibration and exploratory acoustic/ASR fusion.
9. Noise, device, language, severity, and subgroup robustness.
10. Repeated-seed confirmation and error analysis.

The model must not call an expert “articulation,” “phonation,” or “prosody” expert unless a
defined analysis demonstrates that interpretation. Learned routing weights alone do not prove
clinical meaning.

## Security requirements

Candidate code must not use network calls, shell commands, subprocesses, dynamic execution,
unsafe deserialization, arbitrary file access, or secrets. Candidate artifacts may be written
only through `locked/candidate_io.py` into `artifacts/`.

All changes must pass:

- Ruff
- Mypy
- Pytest
- Bandit
- Pip-audit
- CodeQL
- GitHub dependency review
- Secret scanning and push protection

If a vulnerability appears, stop the affected experiment, preserve the evidence, and add it
to the vulnerability register. Do not suppress a scanner finding merely to make CI pass.

## Findings and manuscript rules

Use three evidence states:

- **Proposed:** architecture or hypothesis with no reproduced result.
- **Candidate result:** one controlled run that needs confirmation.
- **Validated finding:** repeated, reproducible evidence that passed security, robustness,
  calibration, and human review.

Only validated findings belong in `research/findings/validated.md` or the final manuscript.
Negative and inconclusive outcomes are valuable research evidence and must remain visible.

The project is not a diagnostic tool. Do not claim clinical validity, treatment benefit, or
deployment readiness without an appropriate clinical study and qualified review.

## Troubleshooting

### No runner is available

Check that the Mac is awake, the Actions runner service is running, the label is exactly
`dysarthria-gpu`, and the runner belongs to this repository. Scheduled jobs remain queued until
the runner reconnects.

Turning the Mac off, losing power, or sleeping in a way that stops the runner terminates the
active job. GitHub does not migrate that process. A later run starts from the beginning unless
the training implementation has saved checkpoints. For unattended local runs, use power,
supported clamshell mode, and macOS settings that prevent sleep while connected to power. A
temporary `caffeinate -dimsu` session can keep the Mac awake, but it cannot keep a powered-off
machine running. Use an always-on or cloud GPU when uninterrupted operation is required.

### No accelerator is detected

Run `verify_runner.py` locally. Confirm Apple Silicon plus a PyTorch MPS build, or use a CUDA
machine. Do not silently fall back to CPU for a claimed GPU experiment.

### The private manifest is rejected

Check the absolute path, file permissions, required columns, pseudonymous speaker IDs, and
whether the path accidentally points to `data/sample_manifest.csv`.

### The model cannot load

Confirm the reviewed immutable model revision is cached locally, the artifact is a supported
`safetensors` file, `trust_remote_code=False` is active, and the model license permits use.

### A metric improves unexpectedly

Stop promotion. Check speaker leakage, duplicate recordings, split hashes, target leakage,
threshold tuning, and whether the validation set was modified. An implausibly large gain is a
diagnostic signal, not proof of a better model.

## Completion checklist

Before calling the project ready:

- the local runner reports MPS or CUDA;
- the real manifest validates without identifiers;
- the model revision and hashes are recorded;
- baseline and ablations run on identical speaker splits;
- all checks pass in GitHub Actions;
- the scheduled workflow completes on the local runner;
- results are reproducible from a clean checkout and private dataset;
- findings distinguish proposal, candidate result, and validated evidence;
- the manuscript and README contain no unsupported metrics or clinical claims.
