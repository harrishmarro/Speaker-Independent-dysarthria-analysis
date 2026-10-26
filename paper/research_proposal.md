# A WavLM-Based Mixture-of-Experts Framework for Speaker-Independent Dysarthria Analysis

**Harish P. S.**

Master of Engineering in Applied Data Science

University of Victoria, Canada

## Abstract

Dysarthria is a motor speech disorder that can affect articulation, phonation, timing,
and intelligibility. This project proposes a speaker-independent research framework that
combines self-supervised WavLM speech representations with a Mixture-of-Experts classifier.
An auxiliary automatic speech recognition analysis measures word error rate when reference
transcripts are available. The evaluation protocol assigns each speaker to exactly one
split to prevent identity leakage and reports classification, calibration, intelligibility,
and subgroup results separately. The framework is a research prototype; experimental and
clinical claims will be made only after reproducible evaluation.

## Research questions

1. Does a Mixture-of-Experts head improve unseen-speaker classification over a comparable
   single-head WavLM classifier?
2. Do routing patterns remain stable across seeds, and do they correspond to measurable
   acoustic characteristics without assigning unsupported clinical meanings to experts?
3. How strongly do acoustic classification outputs correlate with independently computed
   ASR word error rate?
4. How does performance vary across severity labels, languages, recording conditions, and
   demographic groups available under the dataset license?

## Proposed method

Audio is resampled to 16 kHz and processed by a pretrained WavLM encoder. A learned gate
routes each contextual frame representation across multiple feed-forward experts. The
weighted expert outputs are pooled to an utterance representation and classified as control
or dysarthric speech. When a verified reference transcript exists, an ASR pipeline produces
a hypothesis and word error rate. Acoustic and ASR measurements are reported independently;
an exploratory fused score may be analyzed only after its assumptions and thresholds are
declared.

## Evaluation protocol

- Group recordings by pseudonymous speaker ID before splitting.
- Use fixed train, validation, and held-out test speakers.
- Compare MFCC-based, WavLM single-head, and WavLM-MoE systems on identical splits.
- Run ablations for the number of experts, encoder freezing, pooling method, and ASR fusion.
- Report accuracy, balanced accuracy, precision, recall, F1, ROC AUC, calibration, and raw
  confusion counts with uncertainty across repeated seeds.
- Preserve configurations and test predictions so every reported number can be recomputed.

## Limitations and responsible use

Speech is sensitive biometric and health-related data. Raw recordings and participant
identifiers are excluded from this public repository. The prototype is not a diagnostic
device, and exploratory model outputs must not guide treatment or replace assessment by a
qualified speech-language professional.

## Results status

No performance result is claimed in this repository until the experiment can be reproduced
from a documented dataset manifest, saved configuration, and saved predictions. Figures and
numbers from the earlier draft are intentionally excluded pending verification.
