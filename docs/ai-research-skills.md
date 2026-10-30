# AI Research Skills in this repository

The project-local Codex installation contains three selected skills from
[Orchestra Research's AI Research Skills](https://github.com/Orchestra-Research/AI-Research-SKILLs):
autoresearch and two research-ideation skills. Paper-writing skills can be added
when a manuscript is ready for drafting.
The installer lock is `.orchestra-skills.json`; the skill files are in `.codex/skills/`.
They are copied into the repository so a new Codex session opened here can read the
same guidance after this pull request is reviewed and merged.

The requested website welcome page returned HTTP 403 in the cloud workspace. The
installation followed the official repository's documented local `npx` installer,
using package version `1.7.2` and source commit
`773a52944ba4747a18bd4ae9ade53fff041adcbc`.

For research planning, read `.codex/skills/0-autoresearch-skill/SKILL.md`, then
`research-state.yaml`, `findings.md`, `research-log.md`, the current protocol under
`research/designs/`, and the authoritative `experiments/registry.tsv` and
`experiments/search_state.json`. The GitHub workflow is the existing experiment loop;
it requires the private self-hosted runner for real TORGO experiments. A Codex session
does not continue working after it ends, so the skill's suggested Claude Code `/loop`
and OpenClaw cron commands are not active here. Use the repository workflow schedule
and review the registry when resuming a session.

The external skill's general autonomy guidance does not override project privacy,
held-out test controls, or the owner's instruction to request pull-request review
before merging code into `main`. Keep raw audio and participant-level artifacts
outside Git. Do not treat a successful workflow or a high training-speaker score as
an approved model.

To refresh these project-local skills deliberately, review the upstream changes first,
then run the installer on a feature branch with `--local` and open a pull request.
