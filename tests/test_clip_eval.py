import pytest

from dysarthria.clip_eval import k_clip_curve


def test_samples_only_one_recording_per_take() -> None:
    records: list[dict[str, object]] = [
        {"speaker_id": "case", "take_id": "a", "score": -8.0},
        {"speaker_id": "case", "take_id": "a", "score": -7.0},
        {"speaker_id": "case", "take_id": "b", "score": 10.0},
        {"speaker_id": "control", "take_id": "a", "score": -1.0},
        {"speaker_id": "control", "take_id": "b", "score": -1.0},
    ]
    [row] = k_clip_curve(records, {"case": 1, "control": 0}, ks=(2,), draws=40, boots=30)
    assert row["auc_mean"] == row["ba_mean"] == 1.0


def test_excludes_speakers_with_too_few_distinct_takes() -> None:
    records: list[dict[str, object]] = [
        {"speaker_id": "case", "score": 1.0},
        {"speaker_id": "case", "score": 2.0},
        {"speaker_id": "control", "score": -1.0, "take_id": None},
        {"speaker_id": "control", "score": -2.0, "take_id": None},
        {"speaker_id": "one-take", "score": 1.0, "take_id": "a"},
        {"speaker_id": "one-take", "score": 2.0, "take_id": "a"},
    ]
    labels = {"case": 1, "control": 0, "one-take": 1, "no-recordings": 0}
    one, two = k_clip_curve(records, labels, ks=(1, 2), draws=30, boots=20)
    assert (one["k"], one["speakers"], one["excluded"]) == (1, 3, 1)
    assert (two["k"], two["speakers"], two["excluded"]) == (2, 2, 2)


def test_seed_reproduces_draws_and_intervals() -> None:
    labels = {"a": 1, "b": 1, "c": 0, "d": 0}
    scores = {"a": (2.0, -1.0), "b": (-2.0, 1.0), "c": (0.0, -1.0), "d": (-1.0, 1.0)}
    records: list[dict[str, object]] = [
        {"speaker_id": speaker, "score": score}
        for speaker, clips in scores.items()
        for score in clips
    ]
    first = k_clip_curve(records, labels, ks=(1, 2), draws=25, boots=40, seed=17)
    second = k_clip_curve(records, labels, ks=(1, 2), draws=25, boots=40, seed=17)
    assert first == second


def test_easy_data_interval_contains_the_point_estimate() -> None:
    labels = {"a": 1, "b": 1, "c": 0, "d": 0}
    records: list[dict[str, object]] = [
        {"speaker_id": speaker, "score": float(2 * label - 1)}
        for speaker, label in labels.items()
        for _ in range(5)
    ]
    for row in k_clip_curve(records, labels, draws=20, boots=50):
        for metric in ("auc", "ba"):
            assert row[f"{metric}_lo"] <= row[f"{metric}_mean"] <= row[f"{metric}_hi"]
            assert row[f"{metric}_mean"] == 1.0


def test_auc_ties_and_zero_threshold() -> None:
    records: list[dict[str, object]] = [
        {"speaker_id": "case", "score": 0.0},
        {"speaker_id": "control-a", "score": 0.0},
        {"speaker_id": "control-b", "score": -1.0},
    ]
    labels = {"case": 1, "control-a": 0, "control-b": 0}
    [row] = k_clip_curve(records, labels, ks=(1,), draws=3, boots=40)
    assert row["auc_mean"] == row["ba_mean"] == 0.75
    assert row["auc_lo"] == row["ba_lo"] == 0.5
    assert row["auc_hi"] == row["ba_hi"] == 1.0


def test_bootstrap_keeps_all_draws_for_a_sampled_speaker() -> None:
    records: list[dict[str, object]] = [
        {"speaker_id": "case", "score": -2.0},
        {"speaker_id": "case", "score": 1.0},
        {"speaker_id": "control", "score": -1.0},
    ]
    [row] = k_clip_curve(records, {"case": 1, "control": 0}, ks=(1,), draws=200, boots=30)
    assert 0.35 < row["auc_mean"] < 0.65
    for metric in ("auc", "ba"):
        assert row[f"{metric}_lo"] == row[f"{metric}_mean"] == row[f"{metric}_hi"]


def test_rejects_unknown_speakers_and_missing_classes() -> None:
    records: list[dict[str, object]] = [{"speaker_id": "case", "score": 1.0}]
    with pytest.raises(ValueError, match="unknown speaker"):
        k_clip_curve(records, {"other-case": 1, "control": 0}, ks=(1,))
    with pytest.raises(ValueError, match="both classes"):
        k_clip_curve(records, {"case": 1}, ks=(1,))
    with pytest.raises(ValueError, match="k=1.*both classes"):
        k_clip_curve(records, {"case": 1, "control": 0}, ks=(1,))
