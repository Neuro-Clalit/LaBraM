"""Config helper utilities shared by the run scripts.

Currently: CLI ``--set key.sub=value`` override parsing, consumed by every
``run_*.py`` entry point and applied via :meth:`ConfigBase.update`.
"""
import argparse
import json
from typing import List


def add_override_arg(parser: argparse.ArgumentParser) -> None:
    """Register the shared ``--set key.sub=value`` argument on *parser*.

    ``action='extend'`` (not the plain ``nargs='*'`` default) so that repeated
    ``--set`` flags accumulate instead of the last one silently replacing all
    the earlier ones -- spelling one override per flag is the natural way to
    write these commands across multiple lines.
    """
    parser.add_argument('--set', dest='overrides', nargs='*', action='extend',
                        default=[], metavar='KEY=VALUE',
                        help='Dotted-path overrides, e.g. --set trainer.epochs=5; '
                             'repeatable.')


def _coerce_override(s: str):
    """Map a CLI override string to a Python literal.

    Order: JSON object/array (``{...}`` / ``[...]``) -> None -> bool -> int ->
    float -> str. The JSON branch lets dict/list fields be set from the CLI, e.g.
    ``--set sagemaker.weight_s3_uris='{"./checkpoints/labram-base.pth": "s3://b/x.pth"}'``
    or ``--set sagemaker.weight_s3_uris='{}'`` to clear it; a ``{``/``[`` string
    that is not valid JSON falls through to be treated as a plain string.
    """
    stripped = s.strip()
    if stripped[:1] in '{[':
        try:
            return json.loads(stripped)
        except ValueError:
            pass
    if s.lower() == 'none':
        return None
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _join_spaced_overrides(items: List[str]) -> List[str]:
    """Re-join overrides the shell split on spaces around the ``=``.

    ``--set a.b = 1`` reaches us as three argv tokens (``'a.b'``, ``'='``,
    ``'1'``) rather than one; a stray backslash-space before a key (``\\ a.b``)
    leaves the space inside the token. Both are natural to type across a
    multi-line command, so stitch them back into ``key=value`` here.

    A trailing ``=`` only swallows the next token when that token is not itself
    an override -- otherwise ``--set output.output_dir= output.log_dir=`` (two
    keys set to the empty string) would merge into one.
    """
    merged: List[str] = []
    i, n = 0, len(items)
    while i < n:
        tok = items[i].strip()
        i += 1
        if not tok:
            continue
        if '=' not in tok:
            # 'key' '=' 'value' or 'key' '=value'
            nxt = items[i].strip() if i < n else ''
            if not nxt.startswith('='):
                raise ValueError(
                    f'Override must be key=value: {tok!r}. Quote values that '
                    'contain spaces, e.g. --set clearml.task_name="my run".')
            tok += nxt
            i += 1
        if tok.endswith('=') and i < n:
            # 'key=' 'value' -- but leave a genuinely empty value alone.
            nxt = items[i].strip()
            if '=' not in nxt:
                tok += nxt
                i += 1
        merged.append(tok)
    return merged


def parse_overrides(items: List[str]) -> dict:
    """Parse ``--set key.sub=value`` strings into a dict for
    :meth:`ConfigBase.update`."""
    out: dict = {}
    for raw in _join_spaced_overrides(items):
        key, value = raw.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError(f'Override has an empty key: {raw!r}')
        out[key] = _coerce_override(value.strip())
    return out
