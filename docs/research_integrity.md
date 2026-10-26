# Research integrity and privacy

## Ownership

Harish P. S. is the sole author and owner of this project. Names present in an earlier
uploaded paper template are not project authors and are excluded from this repository.
Researchers cited in the proposal remain credited for their published work.

The later editable Word manuscript identifies Harish PS as the sole author and supersedes
the conflicting author block in the earlier PDF export. Its undergraduate affiliation is
historical; current project materials use the University of Victoria affiliation.

## Current evidence boundary

The supplied draft reports accuracy, precision, recall, F1, ROC AUC, confusion-matrix,
t-SNE, correlation, and expert-routing results. The draft does not include the underlying
code, run configuration, predictions, data split, or artifacts required to reproduce those
claims. It also reports precision 0.44, recall 0.80, and F1 0.86 together; those values are
not mathematically consistent under the standard F1 definition.

Accordingly, this repository treats the architecture as **proposed** and all performance
as **unverified until reproduced**. New results must include:

- an immutable experiment configuration;
- speaker IDs assigned to exactly one split;
- class and speaker counts per split;
- saved predictions and labels;
- metric computation from those saved predictions;
- baseline and ablation runs under the same splits;
- uncertainty across repeated seeds or confidence intervals.

## Clinical boundary

The outputs are research signals, not a diagnosis or clinical severity rating. Severity
thresholds require validation against qualified clinical assessment and must not be chosen
from the test set.

## Data handling

- Use pseudonymous speaker identifiers.
- Keep raw audio and protected metadata outside Git.
- Record dataset license, consent scope, retention policy, and access controls.
- Do not publish participant-level predictions that could enable re-identification.
- Document language, recording environment, device, demographic limitations, and likely
  sources of distribution shift.
