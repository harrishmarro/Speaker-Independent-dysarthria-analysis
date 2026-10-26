from dysarthria.manifest import assign_speaker_splits


def test_speakers_never_cross_splits():
    rows = [
        {"audio_path": f"{speaker}-{utterance}.wav", "speaker_id": speaker, "label": label}
        for label, prefix in (("control", "C"), ("dysarthria", "D"))
        for speaker_number in range(1, 11)
        for speaker in [f"{prefix}{speaker_number:02d}"]
        for utterance in range(3)
    ]
    result = assign_speaker_splits(rows, seed=7)
    splits_by_speaker = {}
    for row in result:
        splits_by_speaker.setdefault(row["speaker_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_speaker.values())
    assert {row["split"] for row in result} == {"train", "validation", "test"}


def test_split_is_deterministic():
    rows = [
        {"audio_path": f"{index}.wav", "speaker_id": f"S{index}", "label": "control"}
        for index in range(10)
    ]
    assert assign_speaker_splits(rows, seed=11) == assign_speaker_splits(rows, seed=11)
