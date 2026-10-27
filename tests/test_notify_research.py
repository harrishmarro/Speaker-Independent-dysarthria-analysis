from scripts import notify_research


def test_review_issue_contains_actionable_proposal_without_private_speaker_data(monkeypatch):
    monkeypatch.setattr(notify_research, "recent_results", lambda: [{
        "description": "speaker_knn_5", "balanced_accuracy": "0.500000",
        "f1": "0.000000", "cv_balanced_accuracy": "0.600000",
    }])
    body = notify_research.issue_body(
        "approach_approval_required", "ridge_0_1", "https://github.com/example/actions/runs/1"
    )
    assert "class-balanced ridge" in body
    assert "speaker_knn_5" in body
    assert "ridge_0_1" in body
    assert "0.600000" in body


def test_notification_opens_one_assigned_issue_then_deduplicates(monkeypatch):
    monkeypatch.setattr(notify_research, "recent_results", lambda: [])
    monkeypatch.setenv("GITHUB_REPOSITORY", "harrishmarro/example")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GH_TOKEN", "test-token")
    monkeypatch.setattr(notify_research.sys, "argv", [
        "notify_research.py", "approach_approval_required", "ridge_0_1"
    ])
    issues = []

    def fake_request(method, url, token, payload=None):
        assert token == "test-token"
        if method == "GET":
            return issues
        assert payload["assignees"] == ["harrishmarro"]
        issues.append({"title": payload["title"]})
        return {"html_url": "https://github.com/harrishmarro/example/issues/1"}

    monkeypatch.setattr(notify_research, "request", fake_request)
    notify_research.main()
    notify_research.main()
    assert len(issues) == 1
