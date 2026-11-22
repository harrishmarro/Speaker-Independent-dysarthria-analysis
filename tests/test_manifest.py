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


def test_small_cohort_keeps_a_test_speaker_per_class():
    rows = [
        {"audio_path": f"{prefix}{n}.wav", "speaker_id": f"{prefix}{n}", "label": label}
        for label, prefix in (("control", "C"), ("dysarthria", "D"))
        for n in range(3)
    ]
    result = assign_speaker_splits(rows)
    for label in ("control", "dysarthria"):
        splits = sorted(row["split"] for row in result if row["label"] == label)
        assert splits == ["test", "train", "validation"]


def test_torgo_sized_split_is_unchanged():
    # 7 control / 8 dysarthric speakers: the guard must not alter the existing split.
    rows = [
        {"audio_path": f"{prefix}{n}.wav", "speaker_id": f"{prefix}{n:02d}", "label": label}
        for label, prefix, count in (("control", "C", 7), ("dysarthria", "D", 8))
        for n in range(count)
    ]
    counts = {}
    for row in assign_speaker_splits(rows):
        counts[(row["label"], row["split"])] = counts.get((row["label"], row["split"]), 0) + 1
    assert counts == {("control", "train"): 5, ("control", "validation"): 1,
                      ("control", "test"): 1, ("dysarthria", "train"): 6,
                      ("dysarthria", "validation"): 1, ("dysarthria", "test"): 1}
