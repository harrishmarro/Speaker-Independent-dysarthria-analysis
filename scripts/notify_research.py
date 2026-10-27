#!/usr/bin/env python3
"""Open one GitHub review issue when automated research needs a new approach."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
TITLE_PREFIX = "Research cycle needs review: "


def request(method: str, url: str, token: str, payload: dict[str, object] | None = None):
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        with urlopen(Request(url, data=body, headers=headers, method=method), timeout=30) as reply:
            return json.load(reply)
    except HTTPError as error:
        raise RuntimeError(f"GitHub issue API returned HTTP {error.code}") from error


def recent_results() -> list[dict[str, str]]:
    with (ROOT / "experiments/registry.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [row for row in rows if row["status"] == "candidate_result"][-3:]


def issue_body(reason: str, candidate: str, run_url: str) -> str:
    baseline = json.loads(
        (ROOT / "configs/research/baseline.json").read_text(encoding="utf-8")
    )
    next_step = (
        "Proposed approach: class-balanced ridge classification of the existing acoustic "
        "features, tested at three regularization strengths with speaker-level cross-validation."
        if reason == "approach_approval_required"
        else "Proposed next step: inspect the private speaker-level diagnostics, then choose "
             "and review a new model family before adding it to the candidate catalog."
    )
    lines = [
        "The scheduled research checks are still active, but no new model experiment will start "
        "until the next approach is reviewed.",
        "",
        f"Reason: `{reason}`.",
        f"Accepted baseline: `{baseline['candidate']}` at validation balanced accuracy "
        f"`{baseline['balanced_accuracy']:.6f}` and F1 `{baseline['f1']:.6f}`.",
        f"Next proposed candidate: `{candidate or 'none in the reviewed catalog'}`.",
        next_step,
        "",
        "Recent aggregate results:",
        "",
        "| Candidate | Validation balanced accuracy | F1 | Training-speaker CV balanced accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in recent_results():
        lines.append(
            f"| {row['description']} | {row['balanced_accuracy']} | {row['f1']} | "
            f"{row.get('cv_balanced_accuracy') or 'not recorded'} |"
        )
    lines += [
        "",
        "Recommended next step: review the speaker-level cross-validation results and the "
        "proposed approach, then confirm it in the project chat. The approach is gated in "
        "`autoresearch/approvals.json` until confirmation; no held-out test data is used.",
        "",
        f"Workflow run: {run_url}",
    ]
    return "\n".join(lines)


def main() -> None:
    reason = sys.argv[1] if len(sys.argv) > 1 else "review_required"
    candidate = sys.argv[2] if len(sys.argv) > 2 else ""
    repository = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GH_TOKEN"]
    run_url = f"https://github.com/{repository}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    base_url = f"https://api.github.com/repos/{repository}/issues"
    title = TITLE_PREFIX + (candidate or reason).replace("_", " ")
    page = 1
    while True:
        issues = request("GET", f"{base_url}?state=open&per_page=100&page={page}", token)
        if any(issue["title"] == title and "pull_request" not in issue for issue in issues):
            print("Research review issue is already open")
            return
        if len(issues) < 100:
            break
        page += 1
    owner = repository.split("/", 1)[0]
    issue = request("POST", base_url, token, {
        "title": title,
        "body": issue_body(reason, candidate, run_url),
        "assignees": [owner],
    })
    print(f"Opened research review issue: {issue['html_url']}")


if __name__ == "__main__":
    main()
