"""Git provenance for a run: captured where the code lives, replayed where it runs.

ClearML records the repository, branch, commit and uncommitted changes by looking
for a ``.git`` directory next to the running script. That works for a local run,
but a SageMaker job runs from an extracted source tarball built out of
``git ls-files`` — which deliberately contains no ``.git`` — so the detection
finds nothing and the experiment lands in ClearML with an empty *code* section:
no branch, no commit, no record of the uncommitted edits that produced it.

So the submitting machine captures the metadata (:func:`collect_git_info`), ships
it inside the source tarball under :data:`GIT_INFO_FILENAME`, and the in-container
run replays it onto the ClearML task (:func:`apply_git_info_to_task`, via
``Task.set_script``). A local run has a real ``.git`` and no shipped file, so
ClearML's own detection stays in charge and nothing here applies.

Nothing in here is allowed to fail a training run: every entry point degrades to
``None``/no-op when git is missing, the tree is not a checkout, or ClearML
rejects the metadata.
"""

import json
import os
import re
import subprocess
from typing import Any, Dict, Optional

from labram.utils.logging import get_logger

logger = get_logger(__name__)

# Shipped at the root of the source tarball, so it sits next to the code in
# /opt/ml/code (the container's working directory).
GIT_INFO_FILENAME = 'labram_git_info.json'

# Env var pointing at the file, for callers that relocate it.
GIT_INFO_ENV_VAR = 'LABRAM_GIT_INFO'

# An uncommitted diff is recorded so a run is reproducible, but it is attached to
# every experiment — cap it rather than shipping a pathological blob.
MAX_DIFF_BYTES = 256 * 1024


def _git(root: str, *args: str) -> str:
    """``git <args>`` in *root*, or ``''`` when git/the repo is unavailable."""
    try:
        out = subprocess.run(('git',) + args, cwd=root, check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return ''
    return out.stdout.decode('utf-8', 'replace').strip()


def _sanitize_remote(url: str) -> str:
    """Strip any embedded credentials from a remote URL.

    ``https://user:token@host/org/repo`` would otherwise put a token into the
    experiment record. ``git@host:org/repo`` has no secret and is left alone.
    """
    return re.sub(r'^(\w+://)[^/@]*@', r'\1', url)


def collect_git_info(root: str) -> Optional[Dict[str, Any]]:
    """Describe the git state of the checkout at *root*, or ``None`` if it is not
    a git checkout (or git is unavailable).

    Records the commit, branch, sanitized remote, and — because the submitted
    code is the *working tree*, not the commit — whether it was dirty, which
    files differ, and the diff itself (truncated at :data:`MAX_DIFF_BYTES`).
    """
    commit = _git(root, 'rev-parse', 'HEAD')
    if not commit:
        return None

    status = _git(root, 'status', '--porcelain')
    modified, untracked = [], []
    for line in status.splitlines():
        if not line[:2].strip():
            continue
        path = line[3:].strip()
        (untracked if line.startswith('??') else modified).append(path)

    diff = _git(root, 'diff', 'HEAD')
    truncated = len(diff.encode('utf-8')) > MAX_DIFF_BYTES
    if truncated:
        diff = diff.encode('utf-8')[:MAX_DIFF_BYTES].decode('utf-8', 'ignore')
        diff += '\n... [truncated by LaBraM: diff exceeds '
        diff += f'{MAX_DIFF_BYTES} bytes] ...\n'

    branch = _git(root, 'rev-parse', '--abbrev-ref', 'HEAD')
    return {
        'commit': commit,
        'commit_short': commit[:12],
        'branch': '' if branch == 'HEAD' else branch,   # detached HEAD
        'remote': _sanitize_remote(_git(root, 'config', '--get', 'remote.origin.url')),
        'dirty': bool(status),
        'modified_files': sorted(modified),
        'untracked_files': sorted(untracked),
        'diff': diff,
        'diff_truncated': truncated,
    }


def format_git_summary(info: Optional[Dict[str, Any]]) -> str:
    """One-line human summary for the submit log / banner."""
    if not info:
        return 'no git metadata (not a git checkout)'
    parts = [f"commit {info.get('commit_short') or '?'}"]
    if info.get('branch'):
        parts.append(f"branch {info['branch']}")
    if info.get('dirty'):
        n_mod = len(info.get('modified_files') or ())
        n_new = len(info.get('untracked_files') or ())
        parts.append(f"DIRTY ({n_mod} modified, {n_new} untracked)")
    else:
        parts.append('clean')
    return ', '.join(parts)


def write_git_info(path: str, info: Dict[str, Any]) -> str:
    """Serialize *info* to *path* (returns the path)."""
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(info, handle, indent=2, sort_keys=True)
    return path


def git_info_bytes(info: Dict[str, Any]) -> bytes:
    """*info* as the bytes to embed in the source tarball."""
    return json.dumps(info, indent=2, sort_keys=True).encode('utf-8')


def load_git_info(search_dirs: Optional[list] = None) -> Optional[Dict[str, Any]]:
    """Load the shipped git metadata, or ``None`` when there is none.

    Looks at ``$LABRAM_GIT_INFO`` first, then :data:`GIT_INFO_FILENAME` in each of
    *search_dirs* (default: the working directory, which is ``/opt/ml/code`` in a
    SageMaker container). Returning ``None`` is the normal case for a local run.
    """
    candidates = []
    env_path = os.environ.get(GIT_INFO_ENV_VAR)
    if env_path:
        candidates.append(env_path)
    for directory in (search_dirs if search_dirs is not None else [os.getcwd()]):
        candidates.append(os.path.join(directory, GIT_INFO_FILENAME))

    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding='utf-8') as handle:
                return json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning("Could not read git metadata from %s: %s", path, exc)
    return None


def apply_git_info_to_task(task: Any, info: Optional[Dict[str, Any]] = None,
                           entry_point: str = '') -> bool:
    """Attach shipped git metadata to a ClearML task; ``True`` when applied.

    ``Task.set_script`` fills the experiment's *code* section (repository, branch,
    commit, uncommitted changes) that ClearML could not auto-detect without a
    ``.git``. The same facts are also connected as a ``git`` parameter section so
    they are searchable/sortable in the experiments table. Never raises — tracking
    problems must not fail training.
    """
    if task is None:
        return False
    if info is None:
        info = load_git_info()
    if not info:
        return False

    try:
        task.set_script(
            repository=info.get('remote') or None,
            branch=info.get('branch') or None,
            commit=info.get('commit') or None,
            diff=info.get('diff') or None,
            entry_point=entry_point or None,
        )
    except Exception as exc:  # pragma: no cover - depends on clearml/server
        logger.warning("ClearML set_script (git metadata) failed: %s", exc)
        return False

    try:
        task.connect({
            'commit': info.get('commit', ''),
            'branch': info.get('branch', ''),
            'remote': info.get('remote', ''),
            'dirty': bool(info.get('dirty')),
            'modified_files': ', '.join(info.get('modified_files') or ()),
            'untracked_files': ', '.join(info.get('untracked_files') or ()),
        }, name='git')
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ClearML connect (git metadata) failed: %s", exc)

    logger.info("ClearML code provenance set from shipped git metadata: %s",
                format_git_summary(info))
    return True
