# Dataset intake

The proposed first external data source is the Kaggle dataset [TORGO Audio](https://www.kaggle.com/datasets/pranaykoppula/torgo-audio).
It is a candidate research input only. The repository does not contain the audio.

## Intake gates

Before using this data in an experiment, record the exact Kaggle snapshot, file listing,
license/terms, acquisition date, and checksum in a private data log. Confirm that the
terms permit the planned research and that participant identifiers are not exposed. The
Kaggle API was not reachable from the cloud workspace during setup, so the metadata is
intentionally marked `pending_verification` until it is checked from the runner.

Do not commit audio, extracted features, checkpoints, predictions, or a manifest containing
directly identifying paths. Keep the raw dataset and the usable manifest on the private
runner. The public manifest may contain only pseudonymous speaker IDs and relative, non-
sensitive sample identifiers.

## Private acquisition

Configure Kaggle credentials on the private runner using the Kaggle CLI or an equivalent
secret store. Never put `kaggle.json`, API tokens, or credentials in this repository or in
GitHub Actions logs. A typical runner-side workflow is:

```bash
kaggle datasets download -d pranaykoppula/torgo-audio -p /private/data/kaggle --unzip
```

Inspect the extracted files before creating a manifest. Do not assume the archive layout,
sample rate, labels, or license from the URL alone.

## Manifest requirements

Create a private CSV with the schema shown in the root README:
`audio_path,speaker_id,label,severity,transcript`.

Then run the locked validator and split generator:

```bash
dysarthria-data validate /private/data/torgo/manifest.csv
dysarthria-data split /private/data/torgo/manifest.csv \
  --output /private/data/torgo/splits.csv
```

Before training, audit speaker counts, class/severity balance, duplicate audio hashes,
duration, sample rate, channels, missing transcripts, and speaker-disjoint train/validation/
test membership. Any uncertain label mapping or unavailable transcript must be recorded as
an explicit limitation in `research/findings/inconclusive.md`.

## Dataset registry entry

| Field | Value |
| --- | --- |
| Source | `pranaykoppula/torgo-audio` |
| URL | <https://www.kaggle.com/datasets/pranaykoppula/torgo-audio> |
| Snapshot/checksum | pending runner-side verification |
| License/terms | pending verification; do not publish audio |
| Intended use | speaker-independent research prototype |
| Status | candidate, not yet validated |

