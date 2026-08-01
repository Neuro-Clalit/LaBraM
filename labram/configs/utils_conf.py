"""Config helper utilities shared by the run scripts.

Currently: CLI ``--set key.sub=value`` override parsing, consumed by every
``run_*.py`` entry point and applied via :meth:`ConfigBase.update`.
"""
import argparse
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

    Order: None -> bool -> int -> float -> str.
    """
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


def parse_overrides(items: List[str]) -> dict:
    """Parse ``--set key.sub=value`` strings into a dict for
    :meth:`ConfigBase.update`."""
    out: dict = {}
    for raw in items:
        if '=' not in raw:
            raise ValueError(f'Override must be key=value: {raw!r}')
        key, value = raw.split('=', 1)
        out[key.strip()] = _coerce_override(value.strip())
    return out
