# Negative results

On the fixed two-speaker validation split, ridge alpha 0.1
(`EXP-20261029T070244Z-653fff`) scored 0.000 balanced accuracy versus 0.500 for the
accepted spectral baseline. Ridge alpha 10.0 (`EXP-20261030T015559Z-05f2ba`)
tied the 0.500 baseline; neither met the promotion rule. These are negative outcomes
for this split, not proof that ridge methods fail on dysarthria speech generally.
See `research/research_status.md` and `experiments/registry.tsv`.

## Rhythm features fail the cross-microphone gate (`EXP-20261122T080254Z-0c4e3d`)

Prespecified candidate `rhythm_ridge_1`: four rhythm features per recording (speech
occupancy, pauses per second, mean speech-run length, 2-8 Hz envelope modulation), with
speech activity defined relative to each recording's own speech level. Speaker medians,
class-balanced ridge with alpha 1, no tuning. 13 development speakers; test speakers stayed
locked. Balanced accuracy (BA), with AUC in brackets:

| Model | head→array | array→head |
|---|---|---|
| rhythm, VAD 10 dB | 0.786 (0.690) | **0.548** (0.643) |
| rhythm, VAD 15 dB (primary) | 0.714 (0.690) | **0.690** (0.738) |
| rhythm, VAD 20 dB | 0.702 (0.548) | **0.607** (0.667) |
| rhythm, equalized floor, VAD 15 dB | 0.464 (0.381) | 0.607 (0.548) |
| spectral baseline | 0.500 (0.833) | 0.750 (0.690) |
| channel control (noise floor and SNR) | 0.536 (0.595) | 0.595 (0.762) |
| silence control (non-speech cepstrum) | 0.500 (**0.857**) | 0.500 (**0.857**) |

**Gate:** fails. The rule needs swap BA of at least 0.70 in both directions at every VAD
setting, and at least 0.15 above both controls. `vad10` array→head reaches only 0.548.

On the same microphone, rhythm CV reaches BA 0.702 / AUC 0.738. The spectral baseline
reaches 0.774 / 0.786 in the same run.

**Interpretation:** the silence-only control ranks speakers at AUC 0.857 across microphones
and 0.881 on the same microphone. Its BA is 0.500 only because its decision threshold does
not carry across microphones. Non-speech recording conditions therefore remain the
strongest speaker-ranking cue in this TORGO copy. Neither rhythm nor the spectral baseline
shows a gain over them that 13 speakers could detect.

This is a negative result for these features on TORGO's development speakers. It is not
evidence that rhythm carries no information about dysarthria. As planned, the next step
is frozen transfer to an independent corpus (P3) rather than more tuning on TORGO.

**Gate caveat:** the controls' cross-microphone BA understates them, because their ranking
transfers better than their threshold. Before promoting any future candidate, consider
also requiring an AUC margin over the controls.
