#!/usr/bin/env python
"""Submit cross-validation fold jobs spot-first, falling back to on-demand.

Each (experiment, fold) is submitted with ``scripts/submit_tuh_cohorts.sh``
(``CV_FOLD=k MODE=spot|ondemand``). Rules, re-checked every ``--poll`` seconds:

* spot first, at most ``--spot-slots`` spot jobs in flight (the account's
  ml.g5.2xlarge spot quota);
* a spot job that has not started training ``--spot-wait-h`` hours after its
  submission is stopped and resubmitted on-demand; so is a spot job that fails;
* at most ``--ondemand-slots`` on-demand jobs in flight (counting other
  on-demand jobs of the same instance type);
* a failed on-demand job is retried once.

State (job name, mode, status per fold) is kept in ``--state`` so the
scheduler can be restarted. AWS credentials are the ambient ones (``--profile``,
default ``$AWS_PROFILE``); ``--refresh-cmd`` runs before every poll to renew a
short-lived session -- on an MFA-enforced account the machine's
``aws-mfa-session`` (common repo, machine/bin).

    nohup python scripts/cv_scheduler.py --exp cvbase cvctrl cvaug --folds 5 \
        --profile mfa --refresh-cmd aws-mfa-session &
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import boto3

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TERMINAL = ("Completed", "Failed", "Stopped")


def now():
    return datetime.now(timezone.utc)


def log(msg):
    print(f"[{now():%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)


def submit(exp, fold, mode, python, profile=""):
    env = dict(os.environ, CV_FOLD=str(fold), MODE=mode, TAIL="5", PY=python)
    if profile:
        env["AWS_PROFILE"] = profile
    out = subprocess.run(["bash", "scripts/submit_tuh_cohorts.sh", exp], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=1800).stdout
    names = [line.split()[0] for line in out.splitlines() if line.startswith("labram-") and "(fold=" in line]
    if not names:
        raise RuntimeError(f"submission of {exp} fold {fold} ({mode}) failed:\n{out[-2000:]}")
    return names[-1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", nargs="+", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--instance", default="ml.g5.2xlarge")
    p.add_argument("--spot-slots", type=int, default=10)
    p.add_argument("--ondemand-slots", type=int, default=4)
    p.add_argument("--spot-wait-h", type=float, default=5.0)
    p.add_argument("--poll", type=int, default=300)
    p.add_argument("--state", default="/data/datasets/EEG-public/TUH_EEG/_state/cohort_runs/cv_scheduler_state.json")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE", ""),
                   help="AWS profile for status calls and submissions (default $AWS_PROFILE)")
    p.add_argument("--refresh-cmd", default="",
                   help="command run before every poll to renew credentials, e.g. aws-mfa-session")
    args = p.parse_args(argv)

    state = json.load(open(args.state)) if os.path.exists(args.state) else {}
    for exp in args.exp:
        for k in range(args.folds):
            state.setdefault(f"{exp}/{k}", {"exp": exp, "fold": k, "attempts": []})

    def save():
        json.dump(state, open(args.state, "w"), indent=1)

    while True:
        if args.refresh_cmd:
            subprocess.run(args.refresh_cmd, shell=True, capture_output=True)
        sm = boto3.Session(profile_name=args.profile or None).client("sagemaker")
        # refresh the status of every in-flight attempt
        for key, item in state.items():
            att = item["attempts"][-1] if item["attempts"] else None
            if att and att["status"] not in TERMINAL:
                d = sm.describe_training_job(TrainingJobName=att["job"])
                att["status"] = d["TrainingJobStatus"]
                att["secondary"] = d.get("SecondaryStatus")
                att["started_training"] = bool(d.get("TrainingStartTime"))
                if att["status"] == "Failed":
                    att["failure"] = d.get("FailureReason")
                    log(f"{key}: {att['job']} failed ({att['mode']}): {att.get('failure')}")
                elif att["status"] == "Completed":
                    log(f"{key}: {att['job']} completed ({att['mode']})")
                # spot that never got capacity within the wait window -> on-demand
                age_h = (now() - datetime.fromisoformat(att["submitted"])).total_seconds() / 3600
                if (att["mode"] == "spot" and att["status"] == "InProgress"
                        and not att["started_training"] and age_h > args.spot_wait_h):
                    sm.stop_training_job(TrainingJobName=att["job"])
                    att["status"] = "Stopped"
                    att["failure"] = f"no spot capacity for {age_h:.1f} h"
                    log(f"{key}: {att['job']} stopped after {age_h:.1f} h without spot capacity")
        save()

        def in_flight(mode):
            return sum(1 for it in state.values() for a in it["attempts"][-1:]
                       if a["mode"] == mode and a["status"] not in TERMINAL)
        others = sum(1 for page in sm.get_paginator("list_training_jobs").paginate(StatusEquals="InProgress")
                     for j in page["TrainingJobSummaries"]
                     if not any(a["job"] == j["TrainingJobName"] for it in state.values() for a in it["attempts"])
                     and sm.describe_training_job(TrainingJobName=j["TrainingJobName"])["ResourceConfig"]["InstanceType"] == args.instance
                     and not sm.describe_training_job(TrainingJobName=j["TrainingJobName"]).get("EnableManagedSpotTraining"))

        pending = []
        for key, item in state.items():
            att = item["attempts"][-1] if item["attempts"] else None
            if att is None:
                pending.append((key, "spot"))
            elif att["status"] in ("Failed", "Stopped"):
                n_od = sum(1 for a in item["attempts"] if a["mode"] == "ondemand")
                if n_od < 2:
                    pending.append((key, "ondemand"))
        for key, mode in pending:
            if mode == "spot" and in_flight("spot") >= args.spot_slots:
                continue
            if mode == "ondemand" and in_flight("ondemand") + others >= args.ondemand_slots:
                continue
            item = state[key]
            try:
                job = submit(item["exp"], item["fold"], mode, args.python, args.profile)
            except Exception as exc:                     # quota / transient: retry next poll
                log(f"{key}: {mode} submission failed: {exc}")
                continue
            item["attempts"].append({"job": job, "mode": mode, "status": "InProgress",
                                     "submitted": now().isoformat(), "started_training": False})
            log(f"{key}: submitted {job} ({mode})")
            save()

        done = sum(1 for it in state.values() if it["attempts"] and it["attempts"][-1]["status"] == "Completed")
        given_up = [k for k, it in state.items() if it["attempts"] and it["attempts"][-1]["status"] in ("Failed", "Stopped")
                    and sum(1 for a in it["attempts"] if a["mode"] == "ondemand") >= 2]
        log(f"{done}/{len(state)} completed; spot in flight {in_flight('spot')}, "
            f"on-demand in flight {in_flight('ondemand')} (+{others} other)")
        if done + len(given_up) == len(state):
            log(f"all done; gave up on {given_up}" if given_up else "all done")
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
