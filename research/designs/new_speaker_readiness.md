# First experiment: new-speaker measurement readiness

## Research question

Can an acoustic measure generalize to a previously unseen speaker, without confusing
speech characteristics with microphone or recording-session differences? This is a
prerequisite for the proposed patient application; it is not a severity or therapy
outcome study.

## Completed evidence

The private TORGO manifest contains 17,632 usable recordings from 15 speakers. The
fixed split has 11 training speakers, two validation speakers (one per class), and two
untouched test speakers. The registry records a spectral baseline and three MFCC plus
ridge comparisons. None beat the baseline's validation balanced accuracy of 0.500
or F1 of 0.000. The best ridge training-speaker cross-validation score of 0.917 did
not translate into better validation performance. These results complete the first
feature comparison and show that new-speaker reliability is **not established**.

## Recording-condition audit

Run `uv run python scripts/audit_torgo_conditions.py --manifest
"$HOME/private-data/torgo/splits.clean.csv"` on the private runner. The report is
written under the Git-ignored `artifacts/` directory. It validates that each speaker
has one class and one split, and counts recordings, speakers, microphone types, and
speaker-session combinations by split and class. It contains no speaker IDs or paths.
The scheduled workflow runs the same audit before candidate selection.

TORGO directory names encode speaker, recording session, and microphone. `S01` and
`S02` are recording sessions, **not** therapy visits. Array and head microphones may
record the same underlying speech, so recording-level splits and raw sample counts
must not be treated as independent evidence of generalization.

## Gate for a new-speaker claim

1. Confirm data provenance, license, and a reproducible clean-manifest checksum on
   the private runner. Correct any split or label inconsistency before model fitting.
2. Compare each model with the fixed spectral baseline using held-out-speaker folds.
   Refit preprocessing inside each fold; report speaker-level confusion counts,
   balanced accuracy, F1, and uncertainty across speakers. Keep predictions private.
3. Check microphone and session coverage by class and split. Then quantify whether
   predictions change when a speaker is represented by array versus head microphone
   recordings. Report missing or unpaired conditions rather than assuming stability.
4. Do not use the two fixed test speakers to choose features, thresholds, or models.
   Evaluate them once after a protocol and model are selected and reviewed.
5. Treat the result as exploratory until an independent, sufficiently sized cohort
   supports a reliable estimate. Two validation speakers cannot provide that estimate.

Even a positive classification result would establish neither clinical severity nor
within-patient therapy progress. Those require longitudinal patient recordings and
independent clinician-rated outcomes.
