from autoresearch.decide import decide


def test_only_strict_primary_gain_without_f1_regression_is_promoted():
    assert not decide(0.5, 0.2, 0.5, 0.0)
    assert not decide(0.75, 0.1, 0.5, 0.2)
    assert decide(0.75, 0.2, 0.5, 0.2)
