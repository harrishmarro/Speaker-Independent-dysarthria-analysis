# MFCC plus class-balanced ridge baseline

**Status:** implemented and approved for the private runner; no TORGO result recorded
at design time. This is the next controlled study after the spectral and ridge trials
summarized in `research/research_status.md`.

## Hypothesis

Cepstral features may capture vocal-tract and articulation differences that the current
whole-recording spectral summaries miss. The hypothesis is that an MFCC-based classical
classifier will improve **speaker-level validation balanced accuracy** over the retained
`spectral_equal` baseline, without lower F1, and also improve leave-one-training-speaker-out
cross-validation balanced accuracy without lower F1. A validation tie is a rejection.

## Fixed design

- Use the existing private clean TORGO manifest and fixed speaker-disjoint train,
  validation, and held-out test assignments. Exclude held-out test audio from iteration.
- Require mono, finite 16 kHz input. Center each recording and scale by its maximum
  absolute amplitude, matching the current spectral preprocessing.
- Sample at most 160 uniformly spaced 25 ms frames per recording. Use a Hann window,
  512-point power spectrum, 26 triangular mel filters spanning 0–8 kHz, log filter
  energies, and the first 13 coefficients of an unnormalized DCT-II. Concatenate each
  coefficient's recording mean and standard deviation into a 26-value vector.
- Average recording vectors into one vector per speaker. Standardize features using
  **training speakers only**. Refit standardization inside every leave-one-speaker-out
  fold. Use inverse-class-frequency weighted ridge regression with a fixed zero-margin
  decision threshold and penalties 0.1, 1.0, and 10.0.
- Recompute the retained spectral baseline's training-speaker cross-validation in the
  same run. Keep predictions private and log aggregate results, configuration hash,
  dataset hash, and source commit in `experiments/registry.tsv`.

## Decision and limitations

The controller evaluates one preapproved penalty per cycle and applies the existing
strict validation-plus-cross-validation promotion rule. It records every outcome and
restores the accepted baseline when a candidate does not improve. A better candidate
opens a pull request for review; it is not merged automatically.

The validation set contains only one speaker per class, so balanced accuracy can move
by 0.5 with one changed prediction. These trials are screening evidence. A candidate
that passes this gate still needs repeated-seed or stronger speaker-fold confirmation,
error analysis, and human-reviewed held-out test evaluation before any performance claim.
The fixed threshold is not a calibrated clinical decision boundary.
