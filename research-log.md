# Research decision log

This log records design decisions and reflections. Individual scores and exact provenance
are in `experiments/registry.tsv`; the registry and `experiments/search_state.json` are
authoritative if a scheduled run occurs after this snapshot.

## 2026-10-29 — Reflect on spectral and ridge candidates

The speaker-balanced spectral, nearest-neighbor, and ridge candidates did not beat the
retained spectral baseline on the two-speaker validation split. Ridge alpha 1.0 improved
leave-one-training-speaker-out balanced accuracy to 0.917, but validation tied at 0.500.
The evidence is insufficient to promote it. Chose a new, named MFCC classical baseline
to test whether cepstral features address a feature-representation limitation. The
protocol was committed before the private MFCC run in `0a67d08`.

## 2026-10-30 — First MFCC result

`EXP-20261030T065554Z-208dd7` tested MFCC plus class-balanced ridge with alpha 0.1.
Validation balanced accuracy tied the retained baseline at 0.500; training-speaker CV
balanced accuracy was 0.467 versus 0.817 for the baseline. The controller rejected this
candidate and advanced to MFCC alpha 1.0. This result does not establish that MFCCs are
ineffective generally; two validation speakers are too few for that conclusion.

## Next review

Read the remaining MFCC results when they appear. If the three-penalty family does not
improve the predeclared gates, investigate microphone/session shift and the limitations
of the validation split before expanding the model family. Keep test speakers untouched.
