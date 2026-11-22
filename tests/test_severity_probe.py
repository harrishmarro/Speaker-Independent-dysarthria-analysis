import pytest

from dysarthria.severity_probe import kendall_tau_b, rank_probe


def test_kendall_tau_b_with_ties() -> None:
    # Three concordant, one discordant, and one tie exclusive to each variable.
    assert kendall_tau_b([1, 2, 2, 3], [1, 2, 3, 2]) == pytest.approx(2 / 5)
    # Three concordant, two discordant, and one joint tie.
    assert kendall_tau_b([1, 2, 2, 3], [1, 3, 3, 2]) == pytest.approx(1 / 5)
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    assert kendall_tau_b([1, 1, 1], [1, 2, 3]) == 0.0
    assert kendall_tau_b([], []) == 0.0


def test_perfect_ranking_passes_with_exact_p_value() -> None:
    severity: dict[str, int | None] = {str(index): index for index in range(5)}
    severity["unlabeled"] = None
    scores = {str(index): float(index) for index in range(5)}
    controls = {"constant": dict.fromkeys(scores, 0.0)}

    result = rank_probe(scores, severity, controls)

    assert result["n"] == 5
    assert result["tau"] == result["loo_min"] == 1.0
    assert result["p"] == 1 / 120
    assert result["control_taus"] == {"constant": 0.0}
    assert result["verdict"] == "pass"


def test_exact_p_counts_permutations_with_tied_labels() -> None:
    severity = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 1}
    scores = {speaker: float(label) for speaker, label in severity.items()}

    result = rank_probe(scores, severity, {})

    assert result["tau"] == 1.0
    assert result["p"] == 12 / 120
    assert result["verdict"] == "no severity claim"


def test_perfectly_reversed_candidate_fails_the_one_sided_test() -> None:
    severity = {str(index): index for index in range(5)}
    scores = {speaker: -float(label) for speaker, label in severity.items()}
    result = rank_probe(scores, severity, {})
    assert result["tau"] == result["loo_min"] == -1.0
    assert result["p"] == 1.0
    assert result["verdict"] == "no severity claim"


def test_equally_strong_reversed_control_blocks_claim() -> None:
    severity = {str(index): index for index in range(5)}
    scores = {speaker: float(label) for speaker, label in severity.items()}
    reversed_scores = {speaker: -score for speaker, score in scores.items()}

    result = rank_probe(scores, severity, {"reversed": reversed_scores})

    assert result["tau"] == result["loo_min"] == 1.0
    assert result["p"] == 1 / 120
    assert result["control_taus"] == {"reversed": 1.0}
    assert result["verdict"] == "no severity claim"
    assert "control" in str(result["reason"])


def test_leave_one_speaker_out_checks_effect_of_each_omission() -> None:
    severity = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 1}
    scores = {speaker: float(label) for speaker, label in severity.items()}

    result = rank_probe(scores, severity, {})

    assert result["tau"] == 1.0
    assert result["loo_min"] == 0.0
    assert result["verdict"] == "no severity claim"
    assert "leave-one-speaker-out" in str(result["reason"])


def test_refuses_more_than_nine_labeled_speakers() -> None:
    severity = {str(index): index for index in range(10)}
    scores = {speaker: float(label) for speaker, label in severity.items()}
    with pytest.raises(ValueError, match=r"n <= 9.*Monte Carlo"):
        rank_probe(scores, severity, {})


def test_requires_five_labeled_speakers_and_complete_control_scores() -> None:
    severity = {str(index): index for index in range(5)}
    scores = {speaker: float(label) for speaker, label in severity.items()}
    with pytest.raises(ValueError, match="at least 5"):
        rank_probe(scores, {"0": 0, "1": 1, "2": 2, "3": 3, "4": None}, {})
    with pytest.raises(ValueError, match="control 'incomplete' missing scores"):
        rank_probe(scores, severity, {"incomplete": {"0": 0.0}})
