# Research status (29 October 2026)

This is a record of the public repository and the private-runner results available as of
29 October 2026. The project is a research prototype, not a clinical diagnostic system.
The proposal in `paper/research_proposal.md` describes the intended WavLM and
Mixture-of-Experts (MoE) system; that system has not produced a validated result.

## Data and protocol

- The owner reports a TORGO-derived archive of 17,635 WAV paths from 15 pseudonymous
  speakers. Three recordings were excluded during the private audio audit: two unreadable
  files and one zero-frame file. The resulting private manifest has 17,632 recordings.
  These counts come from the owner's local audit; raw audio and the manifest are not in
  this repository and cannot be independently rechecked here.
- Speaker-disjoint assignment uses 11 training speakers, 2 validation speakers (FC01,
  M03), and 2 held-out test speakers (F04, MC03). The two validation speakers represent
  one control and one dysarthric speaker. Their labels must never be used to fit feature
  scaling, model parameters, or a decision threshold.
- The test speakers have not been used for automated candidate selection. Predictions
  stay on the private runner; the public registry contains aggregate metrics, hashes,
  source commits, and candidate configuration provenance.

## Experiments completed

The accepted comparison baseline is `spectral_equal`, with validation balanced accuracy
0.500 and F1 0.000 (see `experiments/Model_used.tsv`). The public registry records eleven
non-smoke acoustic experiments before the ridge family: a majority-label harness,
speaker-balanced and spectral variants, and nearest-neighbor variants. Their validation
balanced accuracy remained 0.500 with F1 0.000. Older registry rows have incomplete
provenance and lack the later cross-validation fields; their numbers should not be
treated as repeated independent confirmations.

| Candidate | Experiment ID | Validation balanced accuracy | Validation F1 | Training-speaker CV balanced accuracy | Baseline CV balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Ridge, alpha 0.1 | `EXP-20261029T070244Z-653fff` | 0.000 | 0.000 | 0.733 | 0.817 |
| Ridge, alpha 1.0 | `EXP-20261029T220035Z-3b7d29` | 0.500 | 0.000 | 0.917 | 0.817 |
| Ridge, alpha 10.0 | `EXP-20261030T015559Z-05f2ba` | 0.500 | 0.000 | 0.817 | 0.817 |

Each ridge result came from one run with seed 42 and 11 leave-one-training-speaker-out
folds. Alpha 1.0 improved the training-speaker cross-validation score, but it did not
improve validation. Alpha 0.1 regressed on validation and cross-validation. Alpha 10.0
tied the baseline. All three failed the predeclared promotion rule, and the baseline
remains `spectral_equal`. No held-out test score or clinical performance claim follows
from these results.

With one speaker per class in validation, a single error moves balanced accuracy by
0.5. The repeated 0.500 scores therefore contain little information about expected
performance on unseen speakers. The large gap between ridge cross-validation and
validation may reflect speaker or recording-condition shift; the public aggregates do
not identify a cause. Keep this as a hypothesis until privacy-preserving error analysis
and stronger validation evidence are available.

## Current implementation and next study

`train.py` currently computes whole-recording spectral summaries and one prediction per
validation speaker. `src/dysarthria/acoustic_cv.py` supports centroid, nearest-neighbor,
and class-balanced ridge heads. The prior catalog was exhausted at `next_index: 6`.
The MFCC study adds three reviewed classical candidates after that index. The first
result is recorded in the update below; the remaining candidates run separately.

### Update: first MFCC candidate (30 October 2026 UTC)

`EXP-20261030T065554Z-208dd7` tested MFCC features with class-balanced ridge
alpha 0.1. Validation balanced accuracy was 0.500 and F1 was 0.000, tying the
retained baseline. Training-speaker cross-validation balanced accuracy was 0.467
versus 0.817 for the baseline. The controller rejected the candidate and advanced
to index 7. Alpha 1.0 and 10.0 are still pending in this snapshot. No held-out
test result was used.

The next controlled comparison should establish a named MFCC-based classical baseline
before training WavLM. Use the same fixed speaker split, train-only preprocessing,
speaker-level aggregation, leave-one-training-speaker-out cross-validation, and existing
held-out test boundary. Record both validation and cross-validation metrics and keep
private predictions for audit. Then compare a frozen WavLM representation with a linear
head before assessing the proposed MoE head. The WavLM encoder class in
`src/dysarthria/model.py` is architectural scaffolding; it is not connected to the
scheduled experiment harness or a verified pretrained checkpoint.

Before a strong performance claim, confirm the dataset snapshot and license, repeat
promising candidates across seeds or speaker folds, analyze recording/microphone effects,
and arrange human review of a single held-out test evaluation.

### Update: MFCC family complete (30 October 2026 UTC)

The registry now contains all three planned MFCC-plus-ridge trials. Penalties 0.1, 1.0,
and 10.0 each scored 0.500 validation balanced accuracy and 0.000 validation F1.
Training-speaker cross-validation balanced accuracy was 0.467, 0.467, and 0.633,
respectively, versus 0.817 for the retained spectral baseline in each run. None met
the predeclared promotion rule. The accepted model remains `spectral_equal`, and the
finite candidate catalog is exhausted (`next_index: 9`). The earlier snapshot above
predates the second and third MFCC results; the registry is authoritative.

This completes the first acoustic-feature comparison, **not** validation of reliable
speech measurement for a new patient. The next research step is a recording-condition
audit and stronger unseen-speaker evaluation. See
`research/designs/new_speaker_readiness.md`.
