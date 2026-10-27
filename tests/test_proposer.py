from autoresearch.propose import CANDIDATES, next_candidate


def result(candidate: str, score: str) -> dict[str, str]:
    return {
        "mode": "experiment",
        "status": "candidate_result",
        "description": candidate,
        "balanced_accuracy": score,
    }


def test_proposer_waits_for_current_result():
    assert next_candidate(dict(CANDIDATES[0]), [result("unrelated", "0.50")]) is None


def test_proposer_advances_after_weak_result():
    assert next_candidate(dict(CANDIDATES[0]), [result("spectral_equal", "0.50")]) == CANDIDATES[1]


def test_proposer_stops_for_confirmation_or_exhaustion():
    assert next_candidate(dict(CANDIDATES[0]), [result("spectral_equal", "1.00")]) is None
    assert next_candidate(dict(CANDIDATES[-1]), [result("spectral_robust", "0.50")]) is None
