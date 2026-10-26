# Manuscript source review

## Source of record

The editable `Dysarthria Detection Paper.docx` supplied on October 26, 2026 identifies
**Harish PS** as the sole author. The public project must not list any additional project
authors. Published researchers remain credited only through normal citations.

## Details retained

- WavLM self-supervised speech representations
- frame-level Mixture-of-Experts routing
- speaker-disjoint evaluation
- ASR-based word error analysis
- the need to distinguish acoustic detection from severity assessment

## Details corrected before publication

- Update the historical undergraduate affiliation to the current University of Victoria
  Applied Data Science affiliation.
- Treat the framework as proposed until code, dataset manifests, predictions, and run
  configurations reproduce the results.
- Remove unsupported clinical-deployment language.
- Avoid assigning names such as “articulation expert” to learned experts unless a defined
  analysis demonstrates that specialization.
- Recompute all metrics from saved predictions. The reported precision, recall, and F1
  values in the draft are not mutually consistent under the standard F1 definition.
- Validate baseline names, datasets, comparison values, figure captions, and every
  bibliographic entry against the original sources before submission.

## Publication boundary

The original Word manuscript is a private working source. The public repository contains a
corrected proposal and reproducible code. A revised full paper should be generated only
after verified experiments produce traceable results and figures.
