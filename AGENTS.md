# Repository agent guidance

For research planning or synthesis, read the project-local Orchestra autoresearch
skill at `.codex/skills/0-autoresearch-skill/SKILL.md` and the current state in
`research-state.yaml`. Use `experiments/registry.tsv` and
`experiments/search_state.json` as the source of truth for completed experiments.

Keep private TORGO audio, participant-level predictions, and manifests out of Git.
Do not inspect held-out test speakers during candidate selection. Follow the project's
existing security and evaluation gates. Make code changes on a feature branch and
request the owner's pull-request review before merging into `main`.
