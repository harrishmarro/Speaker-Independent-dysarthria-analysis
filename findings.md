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

All three MFCC candidates, with ridge penalties 0.1, 1.0, and 10.0, tied the
validation baseline at 0.500 and scored 0.467, 0.467, and 0.633 respectively in
training-speaker CV, versus 0.817 for the retained spectral baseline. All were
rejected. The finite candidate catalog is exhausted. The MFCC protocol is in
`research/designs/mfcc_baseline.md`.

## Recording-condition evidence

Recording conditions alone predict the label about as well as the models do. On the 13
development speakers, with the test speakers still locked:

- A silence-only control uses only the cepstrum of non-speech frames. It ranks speakers at
  AUC 0.881 in same-microphone cross-validation, and at 0.857 when trained on one
  microphone and scored on the other.
- A channel-only control uses noise floor and SNR. It reaches AUC 0.833 in cross-validation.
- The spectral baseline reaches AUC 0.786 in the same run, below both controls.
- The prespecified rhythm candidate (`EXP-20261122T080254Z-0c4e3d`) fails the
  cross-microphone gate. See `research/findings/negative-results.md`.

On TORGO, no feature tested so far shows a disease-specific signal beyond recording
conditions. Candidates are now promoted only under a speaker-held-out microphone swap.
They must beat both controls on balanced accuracy and on AUC, at every prespecified
setting. See `autoresearch/decide.py` and `controls.py`.

## Lessons and constraints

- Validation has only one speaker per class. A single changed speaker prediction moves
  balanced accuracy by 0.5; the current scores have high uncertainty.
- Raw speech, direct participant identifiers, and private predictions must stay outside
  this public repository.
- Neither a same-microphone cross-validation gain nor a two-speaker validation result is
  promoted. Promotion requires the microphone-swap gate.
- The fixed held-out test speakers have not been used to select a candidate.

## Open questions

- Microphone and session coverage: answered in part. Non-speech recording conditions
  carry the label across both microphones (see above), so earlier cross-validation scores
  cannot be read as disease signal.
- Does any feature beat the recording-condition controls on an independent corpus whose
  case and control speakers were recorded under matched conditions (e.g. UA-Speech,
  PC-GITA)? This is the next step, before WavLM. Mixture-of-Experts is deprioritized.
- Severity cannot be estimated from this archive: it has no transcripts and no
  severity labels.

The full result table and evidence boundary are in `research/research_status.md` and
`experiments/registry.tsv`. The proposed neural architecture has no validated result.
