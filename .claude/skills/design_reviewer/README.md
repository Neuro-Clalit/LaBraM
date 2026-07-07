# `design_reviewer` skill

A vendored copy of the `design_reviewer` skill used to run multi-layered
architectural reviews of LaBraM code. Invoke it as `/design_reviewer <path>`
(e.g. `/design_reviewer labram/`), or describe the task ("review this module
for architectural soundness and maintainability").

## What it does

`SKILL.md` defines a five-layer review protocol (static hygiene, structural
complexity, cognitive load, AI-readiness, testability) and a JSON verdict
schema. `scripts/design_review_metrics.py` is a self-contained CLI that wraps
[`radon`](https://pypi.org/project/radon/) (McCabe cyclomatic complexity +
Maintainability Index) and [`ruff`](https://pypi.org/project/ruff/) (lint),
applies the canonical gates (McCabe warn > 7 / reject > 12, MI < 65), and emits
a text or JSON report. It degrades gracefully — if `radon`/`ruff` are missing it
reports what it skipped instead of failing.

```bash
# text summary
python .claude/skills/design_reviewer/scripts/design_review_metrics.py labram/
# JSON for a correction loop
python .claude/skills/design_reviewer/scripts/design_review_metrics.py --format json labram/
```

## Prerequisite

```bash
pip install radon ruff        # or: pip install -e ".[dev]"
```

## Provenance / re-sync

Vendored from the personal marketplace
[`gugas81/claude-toolbox`](https://github.com/gugas81/claude-toolbox), path
`plugins/research-toolbox/skills/design_reviewer/`. Both `SKILL.md` and
`scripts/design_review_metrics.py` are copied verbatim except that the script
invocation in `SKILL.md` uses this repo-relative path instead of the plugin-only
`$CLAUDE_SKILL_DIR` variable, so it works as a project skill.

To re-sync from the source marketplace, either re-copy those two files, or
install it as a plugin instead:

```
/plugin marketplace add gugas81/claude-toolbox
/plugin install research-toolbox@my-plugins
```
