#!/usr/bin/env python
"""Can SageMaker actually give me these GPU instances right now?

AWS exposes no direct "free capacity" API, so this combines the four signals
that exist into one read-only report (no instances are launched, nothing is
billed):

  1. Service quotas  - the hard per-account ceiling for each instance type,
                       separately for on-demand and spot training. A quota of 0
                       means CreateTrainingJob is rejected outright, no matter
                       how much capacity AWS has.
  2. Spot placement  - EC2's own 1-10 estimate of the odds of getting the
     score              equivalent spot capacity in this region right now.
                       1-3: don't bother waiting; 9-10: should start quickly.
  3. AZ offerings    - which availability zones offer the EC2 family at all
                       (existence in the region, not live capacity).
  4. Recent jobs     - your account's actual outcomes; a run of
                       MaxWaitTimeExceeded / CapacityError rows is the ground
                       truth that beats any estimate.

Usage (profile defaults to the checkout's .aws-profile, like submit_sagemaker):

    python scripts/check_sagemaker_capacity.py
    python scripts/check_sagemaker_capacity.py --types ml.g5.2xlarge,ml.g6.2xlarge
    python scripts/check_sagemaker_capacity.py --profile neuro --region us-east-1
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labram.runs.submit_sagemaker import discover_aws_profile, repo_root  # noqa: E402

DEFAULT_TYPES = ['ml.g5.xlarge', 'ml.g5.2xlarge', 'ml.g5.4xlarge']
RULE = '-' * 76


def ec2_equivalent(ml_type: str) -> str:
    """'ml.g5.2xlarge' -> 'g5.2xlarge' (the EC2 hardware SageMaker runs on)."""
    return ml_type[3:] if ml_type.startswith('ml.') else ml_type


def make_session(profile: str, region: str):
    import boto3
    kwargs = {}
    if profile:
        kwargs['profile_name'] = profile
    if region:
        kwargs['region_name'] = region
    return boto3.Session(**kwargs)


# ---------------------------------------------------------------- 1) quotas


def training_quotas(session, types: List[str]) -> Dict[str, Dict[str, Optional[Dict]]]:
    """{ml_type: {'on_demand': quota|None, 'spot': quota|None}} from Service
    Quotas, where each quota is ``{'value', 'code', 'adjustable'}``. Applied
    values first, falling back to AWS defaults. The code is what a
    ``request-service-quota-increase`` call needs."""
    sq = session.client('service-quotas')
    wanted = {t: {'on_demand': f'{t} for training job usage',
                  'spot': f'{t} for spot training job usage'} for t in types}
    out = {t: {'on_demand': None, 'spot': None} for t in types}

    def scan(op):
        paginator = sq.get_paginator(op)
        for page in paginator.paginate(ServiceCode='sagemaker'):
            for q in page.get('Quotas', []):
                name = q.get('QuotaName', '')
                for t, names in wanted.items():
                    for kind, qname in names.items():
                        if name == qname and out[t][kind] is None:
                            out[t][kind] = {'value': q.get('Value'),
                                            'code': q.get('QuotaCode', ''),
                                            'adjustable': q.get('Adjustable', False)}

    scan('list_service_quotas')                    # account-specific overrides
    if any(v is None for d in out.values() for v in d.values()):
        scan('list_aws_default_service_quotas')    # fill gaps with defaults
    return out


# ---------------------------------------- 2) spot placement + 3) AZ offerings


def spot_scores(session, types: List[str]) -> Dict[str, Optional[int]]:
    """{ec2_type: score 1-10 | None} — EC2's live estimate of spot odds."""
    ec2 = session.client('ec2')
    region = session.region_name
    out = {}
    for t in types:
        try:
            resp = ec2.get_spot_placement_scores(
                InstanceTypes=[t], TargetCapacity=1,
                SingleAvailabilityZone=False, RegionNames=[region])
            scores = [s['Score'] for s in resp.get('SpotPlacementScores', [])]
            out[t] = max(scores) if scores else None
        except Exception as exc:  # AccessDenied / unsupported type
            print(f"  (spot placement score unavailable for {t}: "
                  f"{getattr(exc, 'response', {}).get('Error', {}).get('Code', exc)})")
            out[t] = None
    return out


def az_offerings(session, types: List[str]) -> Dict[str, List[str]]:
    """{ec2_type: [AZs that offer it]} — existence in the region, not capacity."""
    ec2 = session.client('ec2')
    out = {t: [] for t in types}
    try:
        paginator = ec2.get_paginator('describe_instance_type_offerings')
        for page in paginator.paginate(
                LocationType='availability-zone',
                Filters=[{'Name': 'instance-type', 'Values': types}]):
            for o in page.get('InstanceTypeOfferings', []):
                out[o['InstanceType']].append(o['Location'])
    except Exception as exc:
        print(f"  (AZ offerings unavailable: {exc})")
    return {t: sorted(azs) for t, azs in out.items()}


# ------------------------------------------------------------ 4) recent jobs


def recent_jobs(session, limit: int) -> List[Dict]:
    sm = session.client('sagemaker')
    jobs = sm.list_training_jobs(MaxResults=min(limit, 100), SortBy='CreationTime',
                                 SortOrder='Descending')['TrainingJobSummaries']
    rows = []
    for j in jobs[:limit]:
        d = sm.describe_training_job(TrainingJobName=j['TrainingJobName'])
        transitions = d.get('SecondaryStatusTransitions') or [{}]
        rows.append({
            'name': j['TrainingJobName'],
            'status': d['TrainingJobStatus'],
            'secondary': d.get('SecondaryStatus', ''),
            'type': d['ResourceConfig']['InstanceType'],
            'spot': d.get('EnableManagedSpotTraining', False),
            'detail': d.get('FailureReason') or transitions[-1].get('StatusMessage', ''),
        })
    return rows


# ------------------------------------------------------------------- report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--types', default=','.join(DEFAULT_TYPES),
                    help='comma-separated ml.* instance types '
                         f'(default: {",".join(DEFAULT_TYPES)})')
    ap.add_argument('--profile', default='',
                    help="AWS profile (default: the checkout's .aws-profile)")
    ap.add_argument('--region', default='', help='AWS region (default: profile region)')
    ap.add_argument('--jobs', type=int, default=8, help='recent training jobs to show')
    args = ap.parse_args()

    types = [t.strip() for t in args.types.split(',') if t.strip()]
    profile = args.profile or discover_aws_profile(repo_root())[0]
    session = make_session(profile, args.region)
    ident = session.client('sts').get_caller_identity()
    print(f"Account {ident['Account']} ({ident['Arn']}), region {session.region_name}, "
          f"profile {profile or '<default chain>'}")

    print(f"\n{RULE}\n1) SageMaker training quotas — max CONCURRENT instances this account may\n"
          f"   use per type (0 = blocked outright; 1 = one job at a time is fine)\n{RULE}")
    quotas = training_quotas(session, types)
    for t in types:
        q = quotas[t]
        fmt = lambda v: '?' if v is None else f"{int(v['value'])} [{v['code']}]"
        print(f"  {t:<16} on-demand: {fmt(q['on_demand'])}   spot: {fmt(q['spot'])}")
    print("\n  To raise one (needed only for quota 0, parallel jobs, or instance_count>1):\n"
          f"    aws service-quotas request-service-quota-increase \\\n"
          f"        --service-code sagemaker --quota-code <code-above> --desired-value <N> \\\n"
          f"        --profile {profile or 'default'} --region {session.region_name}")

    ec2_types = [ec2_equivalent(t) for t in types]
    print(f"\n{RULE}\n2) Spot placement score, {session.region_name} "
          f"(1-3: capacity is scarce now; 9-10: likely fast)\n{RULE}")
    for t, score in spot_scores(session, ec2_types).items():
        print(f"  {t:<16} {score if score is not None else 'n/a'}")

    print(f"\n{RULE}\n3) Availability zones offering the hardware\n{RULE}")
    for t, azs in az_offerings(session, ec2_types).items():
        print(f"  {t:<16} {', '.join(azs) if azs else 'NOT OFFERED in this region'}")

    print(f"\n{RULE}\n4) Recent training jobs (MaxWaitTimeExceeded = spot never came; "
          f"CapacityError = on-demand shortage)\n{RULE}")
    for r in recent_jobs(session, args.jobs):
        mode = 'spot' if r['spot'] else 'on-demand'
        line = f"  {r['name']:<44} {r['type']:<14} {mode:<9} {r['status']}/{r['secondary']}"
        if r['status'] != 'Completed' and r['detail']:
            line += f"\n      {r['detail'][:110]}"
        print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
