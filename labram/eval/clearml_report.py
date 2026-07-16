# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Command-line entry point for local ClearML experiment analysis:
#
#   python -m labram.eval.clearml_report --task-id <id> --output-dir <dir>
#
# Downloads the experiment into a plain-data snapshot, runs the heuristic
# analysers, and writes ``snapshot.json`` + ``report.md`` (and/or prints the
# Markdown report) for offline inspection. See docs/clearml_local_analysis.md.
# ---------------------------------------------------------

import argparse
from typing import List, Optional

from labram.eval.clearml_analysis import load_and_analyze, render_report


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='labram.eval.clearml_report',
        description="Load a ClearML experiment and analyse it locally, writing a "
                    "Markdown report + JSON snapshot for offline inspection.")
    p.add_argument('--task-id', default=None, help="ClearML task id.")
    p.add_argument('--project-name', default=None,
                   help="Project name (with --task-name) when no id is given.")
    p.add_argument('--task-name', default=None,
                   help="Task name (with --project-name) when no id is given.")
    p.add_argument('--output-dir', default=None,
                   help="Directory to write snapshot.json + report.md into.")
    p.add_argument('--print', action='store_true', dest='print_report',
                   help="Print the Markdown report to stdout.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if not args.task_id and not (args.project_name and args.task_name):
        raise SystemExit("Provide --task-id, or both --project-name and --task-name.")
    snapshot, insights = load_and_analyze(
        task_id=args.task_id, task_name=args.task_name,
        project_name=args.project_name, output_dir=args.output_dir)
    if args.print_report or not args.output_dir:
        print(render_report(snapshot, insights))
    else:
        print(f"Wrote report for task {snapshot.task_id} to {args.output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
