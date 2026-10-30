# Dysarthria research findings

## Research question

Can a model trained on one set of TORGO speakers distinguish dysarthric from control
speech for previously unseen speakers under a speaker-disjoint evaluation protocol?

## Current understanding

No candidate has improved the accepted validation baseline of 0.500 balanced accuracy
and 0.000 F1. The ridge alpha 1.0 candidate reached 0.917 balanced accuracy in
leave-one-training-speaker-out cross-validation, yet tied at 0.500 on validation. This
gap is a warning against treating training-speaker CV as proof of generalization to the
two held-out validation speakers. The cause is unknown; speaker or microphone/session
shift is a hypothesis for later analysis.

## New MFCC evidence

The first MFCC candidate, alpha 0.1 (`EXP-20261030T065554Z-208dd7`), tied the
validation baseline at 0.500 and scored 0.467 in training-speaker CV versus 0.817 for
the baseline. It was rejected. Alpha 1.0 and 10.0 remain in the reviewed candidate
catalog. The MFCC protocol is in `research/designs/mfcc_baseline.md`.

## Lessons and constraints

- Validation has only one speaker per class. A single changed speaker prediction moves
  balanced accuracy by 0.5; the current scores have high uncertainty.
- Raw speech, direct participant identifiers, and private predictions must stay outside
  this public repository.
- A cross-validation improvement without a validation improvement is not promoted.
- The fixed held-out test speakers have not been used to select a candidate.

## Open questions

- Do the remaining MFCC penalties produce a validation improvement that also holds in
  training-speaker CV?
- Are errors associated with speaker identity, microphone, session, or recording
  quality? Private-runner analysis is needed to test this without publishing identities.
- How should the protocol obtain stronger speaker-level validation evidence before a
  frozen WavLM and then Mixture-of-Experts comparison?

The full result table and evidence boundary are in `research/research_status.md` and
`experiments/registry.tsv`. The proposed neural architecture has no validated result.
