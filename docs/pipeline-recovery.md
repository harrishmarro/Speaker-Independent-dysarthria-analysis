# Bounded pipeline recovery

`.github/workflows/pipeline-recovery.yml` runs on a GitHub-hosted runner when CI, Security,
Dependency review, or the research workflow fails or times out. Its default mode uses
reviewed rules to recognize known failures and select a bounded recovery recipe. No OpenAI
API key or usage credits are required. An optional `gpt-6-sol` API key enables model-assisted
classification; API errors fall back to the same rules.

## One-time setup

1. Enable **Settings → Actions → General → Workflow permissions → Allow GitHub Actions to
   create and approve pull requests**. The workflow uses `GITHUB_TOKEN` for retries, issues,
   branches, and PRs. Keep normal branch protection and required review for `main`.
2. Enable GitHub notifications for assigned issues and failed workflow runs if you want email.
3. If you later want GPT-6 SOL diagnosis, add an `OPENAI_API_KEY` Actions secret with access
   to that API model. ChatGPT Plus alone does not provide API usage credits. The key must
   stay in Actions; never add it to the repository or the Mac runner's `.env` file.

Then run **Actions → Pipeline recovery → Run workflow** once. With no API key, its self-check
classifies a synthetic CI error using the reviewed rules. With a key, it also tries GPT-6 SOL
and falls back to the rules if the API is unavailable. It does not modify code, retry a
workflow, or open an issue.

If the self-check reports HTTP 429, inspect the OpenAI API project's usage and billing.
`insufficient_quota` requires credits or a higher project budget. A temporary rate limit
is retried once using `Retry-After` (capped at 20 seconds); if it persists, wait for the
rate-limit window or raise the project's limits before running the self-check again.

If the key is absent or the model is unavailable, the rules can still retry known transient
failures and propose reviewed CI repairs. Unknown failures open an assigned GitHub issue for
non-Security workflows. Security failures remain in the Actions run and its summary, and the
recovery job fails to preserve GitHub's failure alert; no public vulnerability issue is created.
Failures from untrusted forks are excluded from privileged recovery; their ordinary PR checks
remain visible to the author and maintainer.

## Recovery policy

The agent can retry failed jobs **once** when the failure looks transient. It can propose a
repair PR for a failing `main` CI run using only these recipes:

| Recipe | Permitted change |
| --- | --- |
| Ruff autofix | Python files under `src/dysarthria/` and `tests/` |
| Lock refresh | `uv.lock` |
| Missing development dependency | Add allowlisted NumPy or SoundFile to the `dev` extra and refresh `uv.lock` |

It runs the CI checks before pushing a repair branch and explicitly dispatches CI on the
new branch, because pushes made with `GITHUB_TOKEN` do not normally trigger workflows.
A reviewer must merge the resulting PR. If checks or PR creation fail, it assigns a GitHub
issue with the run link and next action. There is no unbounded retry or automatic merge.

Only error excerpts from GitHub-hosted CI and Dependency review may be sent to GPT-6 SOL
when an API key has been configured.
Mac research and Security logs are never sent; the model sees only failed job and step names
for those workflows. The model's response selects a fixed recipe, never a shell command or
unrestricted patch. Untrusted PR code is never executed by the privileged recovery job.

The Mac runner must still be online for research reruns, and manual approval of a new model
family remains separate from pipeline repair.
