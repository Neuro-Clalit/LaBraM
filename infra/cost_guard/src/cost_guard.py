"""Account cost guard: auto-stop idle / long-running EC2, GPU-job alerts, job cap, hygiene report.

One Lambda, three entry modes (``event["mode"]``):

* ``periodic`` (every 15 min) -- EC2 rules + job rules.
* ``jobs`` (SageMaker / Batch submission events) -- job rules only, so the cap reacts in seconds.
* ``hygiene`` (daily) -- report budget leaks: orphaned EBS/EIPs, long-stopped instances,
  unmanaged snapshots, idle SageMaker endpoints/notebooks/apps/warm pools, S3 growth and
  buckets without an incomplete-multipart-upload cleanup rule.

EC2 rules (an instance tagged ``CostGuardSkipUntil=<ISO date>`` is exempt from the
auto-stop rules until that date; notices are still sent):

1. Idle: running >= IDLE_HOURS and every one of the last IDLE_HOURS hours averaged
   CPU < IDLE_CPU_PCT, and (when the CloudWatch agent publishes GPU metrics) GPU
   utilisation never reached IDLE_GPU_PCT -> stop.
2. Running >= NOTIFY_RUNTIME_HOURS -> one email per start.
3. Running >= MAX_RUNTIME_HOURS -> stop (warning email WARN_BEFORE_STOP_HOURS earlier).

Job rules (SageMaker training + processing, AWS Batch):

4. More than GPU_JOB_ALERT active GPU jobs -> email (at most every ALERT_REPEAT_HOURS).
5. More than MAX_ACTIVE_JOBS active jobs -> the newest ones beyond the cap are stopped.

``DRY_RUN=true`` (env, or ``event["dry_run"]``) logs and emails what *would* happen but
never stops anything.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EXEMPT_TAG = "CostGuardSkipUntil"
RUNTIME_NOTICE_TAG = "CostGuardRuntimeNotice"
STOP_WARNING_TAG = "CostGuardStopWarning"
# Instances whose lifecycle is owned by another service: stopping them fights that service
# (an ASG would replace the "unhealthy" node), so they only get notices.
MANAGED_TAG_PREFIXES = ("aws:autoscaling:", "aws:ec2spot:", "aws:batch:", "eks:", "AWSBatchServiceTag")
GPU_INSTANCE_RE = re.compile(r"^(ml\.)?(g\d|p\d|gr\d)", re.IGNORECASE)
ACTIVE_BATCH_STATUSES = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")


@dataclass(frozen=True)
class Config:
    topic_arn: str = ""
    dry_run: bool = True
    idle_hours: int = 4
    idle_cpu_pct: float = 1.0
    idle_gpu_pct: float = 5.0
    notify_runtime_hours: float = 36
    max_runtime_hours: float = 72
    warn_before_stop_hours: float = 6
    gpu_job_alert: int = 16
    max_active_jobs: int = 128
    alert_repeat_hours: float = 6
    stopped_instance_days: int = 14
    unmanaged_snapshot_days: int = 30
    s3_total_alert_gb: float = 400
    s3_daily_growth_alert_gb: float = 25
    state_param: str = "/cost-guard/state"

    @classmethod
    def from_env(cls, overrides: dict | None = None) -> "Config":
        env = os.environ
        kwargs = {}
        for name, f in cls.__dataclass_fields__.items():
            raw = env.get(name.upper())
            if raw is None:
                continue
            kwargs[name] = _coerce(raw, type(f.default))
        for name, value in (overrides or {}).items():
            if name in cls.__dataclass_fields__:
                kwargs[name] = _coerce(value, type(cls.__dataclass_fields__[name].default))
        return cls(**kwargs)


def _coerce(value, kind):
    if kind is bool:
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes")
    return kind(value)


# --------------------------------------------------------------------------------------
# Pure decision logic (unit-tested without AWS)
# --------------------------------------------------------------------------------------

@dataclass
class InstanceFacts:
    instance_id: str
    name: str
    instance_type: str
    launch_time: dt.datetime
    tags: dict
    lifecycle: str | None = None
    # 5-minute CPU averages and GPU maxima as (timestamp, value) pairs.
    cpu: list = field(default_factory=list)
    gpu: list = field(default_factory=list)


@dataclass
class Decision:
    stop_reason: str | None = None
    notices: list = field(default_factory=list)  # (tag_key, subject, body)
    info: list = field(default_factory=list)


def hours_running(launch_time: dt.datetime, now: dt.datetime) -> float:
    return (now - launch_time).total_seconds() / 3600.0


def exempt_until(tags: dict, now: dt.datetime) -> dt.datetime | None:
    """Return the exemption end if ``CostGuardSkipUntil`` is set to a future date."""
    raw = tags.get(EXEMPT_TAG)
    if not raw:
        return None
    try:
        until = dt.datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if until.tzinfo is None:
        until = until.replace(tzinfo=dt.timezone.utc)
    return until if until > now else None


def hourly_buckets(series: list, now: dt.datetime, hours: int) -> list[list[float]]:
    """Group (timestamp, value) samples into ``hours`` one-hour windows ending at ``now``."""
    buckets: list[list[float]] = [[] for _ in range(hours)]
    for ts, value in series:
        idx = int((now - ts).total_seconds() // 3600)
        if 0 <= idx < hours:
            buckets[idx].append(value)
    return buckets


def idle_verdict(facts: InstanceFacts, now: dt.datetime, cfg: Config) -> tuple[bool, str]:
    """True when CPU (and GPU, if measured) stayed below threshold for every hour."""
    if hours_running(facts.launch_time, now) < cfg.idle_hours:
        return False, "running less than the idle window"
    cpu_hours = hourly_buckets(facts.cpu, now, cfg.idle_hours)
    # Basic monitoring gives 12 samples/hour; demand half so a metrics gap never reads as idle.
    if any(len(b) < 6 for b in cpu_hours):
        return False, "incomplete CPU data"
    cpu_means = [sum(b) / len(b) for b in cpu_hours]
    if max(cpu_means) >= cfg.idle_cpu_pct:
        return False, f"CPU busy (max hourly avg {max(cpu_means):.2f}%)"
    gpu_values = [v for b in hourly_buckets(facts.gpu, now, cfg.idle_hours) for v in b]
    if gpu_values and max(gpu_values) >= cfg.idle_gpu_pct:
        return False, f"GPU busy (peak {max(gpu_values):.0f}%)"
    gpu_note = f"GPU peak {max(gpu_values):.0f}%" if gpu_values else "no GPU metrics"
    return True, (
        f"CPU hourly averages {', '.join(f'{m:.2f}%' for m in reversed(cpu_means))} "
        f"(< {cfg.idle_cpu_pct}%) over the last {cfg.idle_hours} h; {gpu_note}"
    )


def is_service_managed(facts: InstanceFacts) -> bool:
    if facts.lifecycle == "spot":
        return True
    return any(k.startswith(MANAGED_TAG_PREFIXES) for k in facts.tags)


def decide_instance(facts: InstanceFacts, now: dt.datetime, cfg: Config) -> Decision:
    d = Decision()
    hours = hours_running(facts.launch_time, now)
    start_id = str(int(facts.launch_time.timestamp()))
    label = f"{facts.name or facts.instance_id} ({facts.instance_id}, {facts.instance_type})"
    exempt = exempt_until(facts.tags, now)
    managed = is_service_managed(facts)
    can_stop = exempt is None and not managed
    why_not = (
        f"exempt via {EXEMPT_TAG} until {exempt:%Y-%m-%d %H:%M} UTC" if exempt
        else "lifecycle owned by another service (spot/ASG/Batch/EKS)" if managed else ""
    )

    # Rule 3: hard runtime cap.
    if hours >= cfg.max_runtime_hours:
        if can_stop:
            d.stop_reason = f"running {hours:.1f} h >= {cfg.max_runtime_hours:g} h limit"
        elif facts.tags.get(STOP_WARNING_TAG) != start_id:
            d.notices.append((STOP_WARNING_TAG, f"EC2 {facts.name or facts.instance_id} past "
                              f"{cfg.max_runtime_hours:g} h, NOT stopped",
                              f"{label} has run {hours:.1f} h but was not stopped: {why_not}."))
    elif hours >= cfg.max_runtime_hours - cfg.warn_before_stop_hours and can_stop \
            and facts.tags.get(STOP_WARNING_TAG) != start_id:
        stop_at = facts.launch_time + dt.timedelta(hours=cfg.max_runtime_hours)
        d.notices.append((STOP_WARNING_TAG, f"EC2 {facts.name or facts.instance_id} will be stopped "
                          f"at {stop_at:%Y-%m-%d %H:%M} UTC",
                          f"{label} has run {hours:.1f} h and will be stopped at "
                          f"{stop_at:%Y-%m-%d %H:%M} UTC ({cfg.max_runtime_hours:g} h limit).\n"
                          f"To keep it running, tag it {EXEMPT_TAG}=<ISO date>, e.g.\n"
                          f"  aws ec2 create-tags --resources {facts.instance_id} --tags "
                          f"Key={EXEMPT_TAG},Value={(now + dt.timedelta(days=2)):%Y-%m-%dT%H:%MZ}\n"
                          "Instance-store disks (e.g. /opt/dlami/nvme) are wiped on stop."))

    # Rule 1: idle.
    if d.stop_reason is None:
        idle, detail = idle_verdict(facts, now, cfg)
        d.info.append(detail)
        if idle and can_stop:
            d.stop_reason = f"idle: {detail}"

    # Rule 2: long-running notice (once per start).
    if hours >= cfg.notify_runtime_hours and facts.tags.get(RUNTIME_NOTICE_TAG) != start_id \
            and d.stop_reason is None:
        extra = f"\nAuto-stop is disabled for it: {why_not}." if why_not else (
            f"\nIt will be stopped automatically at "
            f"{facts.launch_time + dt.timedelta(hours=cfg.max_runtime_hours):%Y-%m-%d %H:%M} UTC.")
        d.notices.append((RUNTIME_NOTICE_TAG, f"EC2 {facts.name or facts.instance_id} running "
                          f"{hours:.0f} h", f"{label} has been running {hours:.1f} h "
                          f"(since {facts.launch_time:%Y-%m-%d %H:%M} UTC).{extra}"))
    return d


@dataclass
class Job:
    kind: str  # sagemaker-training | sagemaker-processing | batch
    name: str
    created: dt.datetime
    instance_type: str = ""
    instance_count: int = 1
    gpu: bool = False
    queue: str = ""
    job_id: str = ""


def is_gpu_instance(instance_type: str) -> bool:
    return bool(GPU_INSTANCE_RE.match(instance_type or ""))


def jobs_over_cap(jobs: list[Job], cap: int) -> list[Job]:
    """The newest jobs beyond ``cap`` (the oldest ``cap`` keep running)."""
    return sorted(jobs, key=lambda j: j.created)[cap:]


def gpu_alert_due(gpu_jobs: int, state: dict, now: dt.datetime, cfg: Config) -> bool:
    if gpu_jobs <= cfg.gpu_job_alert:
        return False
    last = state.get("gpu_alert_at")
    if not last:
        return True
    elapsed = (now - dt.datetime.fromisoformat(last)).total_seconds() / 3600
    return elapsed >= cfg.alert_repeat_hours or gpu_jobs > int(state.get("gpu_alert_count", 0)) * 1.5


def parse_stop_time(reason: str) -> dt.datetime | None:
    """EC2 StateTransitionReason looks like 'User initiated (2026-09-01 12:00:00 GMT)'."""
    m = re.search(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) GMT\)", reason or "")
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=dt.timezone.utc) if m else None


# --------------------------------------------------------------------------------------
# AWS plumbing
# --------------------------------------------------------------------------------------

def _clients():
    import boto3
    names = ("ec2", "cloudwatch", "sagemaker", "batch", "sns", "ssm", "s3")
    return {n: boto3.client(n) for n in names}


def notify(c, cfg: Config, subject: str, body: str) -> None:
    prefix = "[cost-guard DRY-RUN] " if cfg.dry_run else "[cost-guard] "
    logger.info("NOTIFY %s%s\n%s", prefix, subject, body)
    if cfg.topic_arn:
        c["sns"].publish(TopicArn=cfg.topic_arn, Subject=(prefix + subject)[:100], Message=body)


def _metric_series(c, queries: list[dict], start: dt.datetime, end: dt.datetime) -> dict:
    out: dict[str, list] = {q["Id"]: [] for q in queries}
    for i in range(0, len(queries), 500):
        kwargs = {"MetricDataQueries": queries[i:i + 500], "StartTime": start, "EndTime": end}
        while True:
            resp = c["cloudwatch"].get_metric_data(**kwargs)
            for r in resp["MetricDataResults"]:
                out[r["Id"]].extend(zip(r["Timestamps"], r["Values"]))
            if not resp.get("NextToken"):
                break
            kwargs["NextToken"] = resp["NextToken"]
    return out


def _metric_query(qid: str, namespace: str, name: str, instance_id: str, stat: str) -> dict:
    return {"Id": qid, "ReturnData": True, "MetricStat": {
        "Metric": {"Namespace": namespace, "MetricName": name,
                   "Dimensions": [{"Name": "InstanceId", "Value": instance_id}]},
        "Period": 300, "Stat": stat}}


def gather_instances(c, now: dt.datetime, cfg: Config) -> list[InstanceFacts]:
    facts = []
    pages = c["ec2"].get_paginator("describe_instances").paginate(
        Filters=[{"Name": "instance-state-name", "Values": ["running"]}])
    for page in pages:
        for res in page["Reservations"]:
            for inst in res["Instances"]:
                tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
                facts.append(InstanceFacts(
                    instance_id=inst["InstanceId"], name=tags.get("Name", ""),
                    instance_type=inst["InstanceType"], launch_time=inst["LaunchTime"], tags=tags,
                    lifecycle=inst.get("InstanceLifecycle")))
    if not facts:
        return facts
    queries = []
    for i, f in enumerate(facts):
        queries.append(_metric_query(f"cpu{i}", "AWS/EC2", "CPUUtilization", f.instance_id, "Average"))
        queries.append(_metric_query(f"gpu{i}", "CWAgent", "nvidia_smi_utilization_gpu",
                                     f.instance_id, "Maximum"))
    series = _metric_series(c, queries, now - dt.timedelta(hours=cfg.idle_hours), now)
    for i, f in enumerate(facts):
        f.cpu, f.gpu = series[f"cpu{i}"], series[f"gpu{i}"]
    return facts


def run_ec2_rules(c, cfg: Config, now: dt.datetime) -> list[dict]:
    results = []
    for f in gather_instances(c, now, cfg):
        d = decide_instance(f, now, cfg)
        start_id = str(int(f.launch_time.timestamp()))
        entry = {"instance": f.instance_id, "hours": round(hours_running(f.launch_time, now), 2),
                 "stop": d.stop_reason, "notices": [n[1] for n in d.notices], "info": d.info}
        results.append(entry)
        for tag_key, subject, body in d.notices:
            notify(c, cfg, subject, body)
            if not cfg.dry_run:
                c["ec2"].create_tags(Resources=[f.instance_id], Tags=[{"Key": tag_key, "Value": start_id}])
        if d.stop_reason:
            label = f"{f.name or f.instance_id} ({f.instance_id}, {f.instance_type})"
            body = (f"Stopping {label}: {d.stop_reason}.\nStarted {f.launch_time:%Y-%m-%d %H:%M} UTC. "
                    f"Restart with: aws ec2 start-instances --instance-ids {f.instance_id}\n"
                    f"Exempt it next time with tag {EXEMPT_TAG}=<ISO date>.")
            if cfg.dry_run:
                notify(c, cfg, f"Would stop EC2 {f.name or f.instance_id}", body)
                continue
            try:
                c["ec2"].stop_instances(InstanceIds=[f.instance_id])
                notify(c, cfg, f"Stopped EC2 {f.name or f.instance_id}", body)
            except Exception as exc:  # e.g. stop protection enabled
                entry["error"] = str(exc)
                notify(c, cfg, f"FAILED to stop EC2 {f.name or f.instance_id}", f"{body}\n\nError: {exc}")
    return results


def gather_jobs(c) -> list[Job]:
    sm = c["sagemaker"]
    jobs: list[Job] = []
    for page in sm.get_paginator("list_training_jobs").paginate(StatusEquals="InProgress"):
        for s in page["TrainingJobSummaries"]:
            desc = sm.describe_training_job(TrainingJobName=s["TrainingJobName"])
            rc = desc.get("ResourceConfig", {})
            groups = rc.get("InstanceGroups") or [
                {"InstanceType": rc.get("InstanceType", ""), "InstanceCount": rc.get("InstanceCount", 1)}]
            types = [g["InstanceType"] for g in groups]
            jobs.append(Job("sagemaker-training", s["TrainingJobName"], s["CreationTime"],
                            ",".join(types), sum(g["InstanceCount"] for g in groups),
                            any(is_gpu_instance(t) for t in types)))
    for page in sm.get_paginator("list_processing_jobs").paginate(StatusEquals="InProgress"):
        for s in page["ProcessingJobSummaries"]:
            desc = sm.describe_processing_job(ProcessingJobName=s["ProcessingJobName"])
            cc = desc.get("ProcessingResources", {}).get("ClusterConfig", {})
            itype = cc.get("InstanceType", "")
            jobs.append(Job("sagemaker-processing", s["ProcessingJobName"], s["CreationTime"],
                            itype, cc.get("InstanceCount", 1), is_gpu_instance(itype)))
    batch = c["batch"]
    for qpage in batch.get_paginator("describe_job_queues").paginate():
        for q in qpage["jobQueues"]:
            ids = []
            for status in ACTIVE_BATCH_STATUSES:
                for p in batch.get_paginator("list_jobs").paginate(jobQueue=q["jobQueueArn"], jobStatus=status):
                    ids += [j["jobId"] for j in p["jobSummaryList"]]
            for i in range(0, len(ids), 100):
                for j in batch.describe_jobs(jobs=ids[i:i + 100])["jobs"]:
                    reqs = j.get("container", {}).get("resourceRequirements", [])
                    for node in j.get("nodeProperties", {}).get("nodeRangeProperties", []):
                        reqs += node.get("container", {}).get("resourceRequirements", [])
                    jobs.append(Job("batch", j["jobName"],
                                    dt.datetime.fromtimestamp(j["createdAt"] / 1000, dt.timezone.utc),
                                    gpu=any(r.get("type") == "GPU" for r in reqs),
                                    queue=q["jobQueueName"], job_id=j["jobId"]))
    return jobs


def _load_state(c, cfg: Config) -> dict:
    try:
        return json.loads(c["ssm"].get_parameter(Name=cfg.state_param)["Parameter"]["Value"])
    except c["ssm"].exceptions.ParameterNotFound:
        return {}


def _save_state(c, cfg: Config, state: dict) -> None:
    c["ssm"].put_parameter(Name=cfg.state_param, Value=json.dumps(state), Type="String", Overwrite=True)


def _stop_job(c, job: Job) -> None:
    if job.kind == "sagemaker-training":
        c["sagemaker"].stop_training_job(TrainingJobName=job.name)
    elif job.kind == "sagemaker-processing":
        c["sagemaker"].stop_processing_job(ProcessingJobName=job.name)
    else:
        c["batch"].terminate_job(jobId=job.job_id, reason="cost-guard: active job cap exceeded")


def run_job_rules(c, cfg: Config, now: dt.datetime) -> dict:
    jobs = gather_jobs(c)
    gpu_jobs = [j for j in jobs if j.gpu]
    summary = {"active_jobs": len(jobs), "gpu_jobs": len(gpu_jobs),
               "gpu_instances": sum(j.instance_count for j in gpu_jobs), "stopped": []}

    def table(js):
        return "\n".join(f"  {j.created:%m-%d %H:%M} {j.kind:21} {j.instance_type or j.queue:18} "
                         f"x{j.instance_count} {j.name}" for j in sorted(js, key=lambda j: j.created))

    over = jobs_over_cap(jobs, cfg.max_active_jobs)
    if over:
        for j in over:
            if not cfg.dry_run:
                try:
                    _stop_job(c, j)
                    summary["stopped"].append(j.name)
                except Exception as exc:
                    logger.exception("failed to stop %s: %s", j.name, exc)
        notify(c, cfg, f"{len(jobs)} active jobs > cap {cfg.max_active_jobs}: "
               f"{'would stop' if cfg.dry_run else 'stopped'} {len(over)}",
               f"Active jobs: {len(jobs)} (cap {cfg.max_active_jobs}). The newest {len(over)} were "
               f"{'selected (dry run)' if cfg.dry_run else 'stopped'}:\n{table(over)}")

    state = _load_state(c, cfg)
    if gpu_alert_due(len(gpu_jobs), state, now, cfg):
        notify(c, cfg, f"{len(gpu_jobs)} GPU jobs active (> {cfg.gpu_job_alert})",
               f"{len(gpu_jobs)} GPU jobs on {summary['gpu_instances']} GPU instances are active "
               f"(alert threshold {cfg.gpu_job_alert}).\n{table(gpu_jobs)}")
        state.update(gpu_alert_at=now.isoformat(), gpu_alert_count=len(gpu_jobs))
        _save_state(c, cfg, state)
    elif len(gpu_jobs) <= cfg.gpu_job_alert and state.get("gpu_alert_at"):
        state.pop("gpu_alert_at", None)
        state.pop("gpu_alert_count", None)
        _save_state(c, cfg, state)
    return summary


# ----------------------------------- hygiene ------------------------------------------

def _s3_sizes(c, now: dt.datetime) -> dict[str, tuple[float, float]]:
    """bucket -> (latest GB, GB one day earlier), summed over storage classes."""
    metrics = c["cloudwatch"].list_metrics(Namespace="AWS/S3", MetricName="BucketSizeBytes")["Metrics"]
    queries = [{"Id": f"s{i}", "ReturnData": True, "MetricStat": {
        "Metric": m, "Period": 86400, "Stat": "Average"}} for i, m in enumerate(metrics)]
    if not queries:
        return {}
    series = _metric_series(c, queries, now - dt.timedelta(days=4), now)
    sizes: dict[str, list[float]] = {}
    for i, m in enumerate(metrics):
        bucket = next(d["Value"] for d in m["Dimensions"] if d["Name"] == "BucketName")
        points = sorted(series[f"s{i}"], reverse=True)
        latest = points[0][1] if points else 0.0
        prev = points[1][1] if len(points) > 1 else latest
        cur = sizes.setdefault(bucket, [0.0, 0.0])
        cur[0] += latest / 1e9
        cur[1] += prev / 1e9
    return {b: (v[0], v[1]) for b, v in sizes.items()}


def _has_mpu_cleanup(c, bucket: str) -> bool | None:
    try:
        rules = c["s3"].get_bucket_lifecycle_configuration(Bucket=bucket)["Rules"]
    except Exception as exc:
        if "NoSuchLifecycleConfiguration" in str(exc):
            return False
        return None  # no permission / other account's bucket
    return any(r.get("Status") == "Enabled" and "AbortIncompleteMultipartUpload" in r for r in rules)


def run_hygiene(c, cfg: Config, now: dt.datetime) -> list[str]:
    ec2, sm = c["ec2"], c["sagemaker"]
    findings: list[str] = []

    for v in ec2.describe_volumes(Filters=[{"Name": "status", "Values": ["available"]}])["Volumes"]:
        findings.append(f"Unattached EBS volume {v['VolumeId']} {v['Size']} GB {v['VolumeType']} "
                        f"(created {v['CreateTime']:%Y-%m-%d}) ~${v['Size'] * 0.08:.0f}/mo")
    for a in ec2.describe_addresses()["Addresses"]:
        if not a.get("AssociationId"):
            findings.append(f"Unassociated Elastic IP {a.get('PublicIp')} ~$3.6/mo")
    for n in ec2.describe_nat_gateways(Filter=[{"Name": "state", "Values": ["available"]}])["NatGateways"]:
        findings.append(f"NAT gateway {n['NatGatewayId']} in {n['VpcId']} ~$33/mo + data")

    stopped = ec2.describe_instances(Filters=[{"Name": "instance-state-name", "Values": ["stopped"]}])
    for res in stopped["Reservations"]:
        for inst in res["Instances"]:
            since = parse_stop_time(inst.get("StateTransitionReason", ""))
            if since and (now - since).days >= cfg.stopped_instance_days:
                name = next((t["Value"] for t in inst.get("Tags", []) if t["Key"] == "Name"), "")
                findings.append(f"Instance {name} {inst['InstanceId']} stopped {(now - since).days} days "
                                "(its EBS volumes are still billed)")

    old = now - dt.timedelta(days=cfg.unmanaged_snapshot_days)
    snaps = [s for p in ec2.get_paginator("describe_snapshots").paginate(OwnerIds=["self"])
             for s in p["Snapshots"]]
    unmanaged = [s for s in snaps if s["StartTime"] < old
                 and not any(t["Key"].startswith("aws:dlm:") for t in s.get("Tags", []))]
    if unmanaged:
        findings.append(f"{len(unmanaged)} snapshots older than {cfg.unmanaged_snapshot_days} days not "
                        f"managed by DLM (source volumes total {sum(s['VolumeSize'] for s in unmanaged)} GB)")

    for e in sm.list_endpoints(StatusEquals="InService")["Endpoints"]:
        findings.append(f"SageMaker endpoint {e['EndpointName']} InService (billed hourly)")
    for nb in sm.list_notebook_instances(StatusEquals="InService")["NotebookInstances"]:
        findings.append(f"SageMaker notebook {nb['NotebookInstanceName']} {nb['InstanceType']} InService")
    for app in sm.list_apps()["Apps"]:
        if app.get("Status") == "InService" and app.get("AppType") != "JupyterServer":
            findings.append(f"SageMaker Studio app {app['AppName']} ({app['AppType']}) InService")
    for j in sm.list_training_jobs(WarmPoolStatusEquals="Available")["TrainingJobSummaries"]:
        findings.append(f"SageMaker warm pool kept alive by {j['TrainingJobName']} (billed while Available)")

    sizes = _s3_sizes(c, now)
    total = sum(v[0] for v in sizes.values())
    if total >= cfg.s3_total_alert_gb:
        findings.append(f"S3 total {total:.0f} GB >= {cfg.s3_total_alert_gb:g} GB alert level")
    for bucket, (cur, prev) in sorted(sizes.items()):
        if cur - prev >= cfg.s3_daily_growth_alert_gb:
            findings.append(f"S3 bucket {bucket} grew {cur - prev:.0f} GB in a day (now {cur:.0f} GB)")
    for b in c["s3"].list_buckets()["Buckets"]:
        if _has_mpu_cleanup(c, b["Name"]) is False:
            findings.append(f"S3 bucket {b['Name']} has no AbortIncompleteMultipartUpload lifecycle rule")

    if findings:
        notify(c, cfg, f"Daily cost hygiene: {len(findings)} finding(s)",
               "Possible budget leaks:\n- " + "\n- ".join(findings)
               + f"\n\nS3 storage: {total:.0f} GB total\n" + "\n".join(
                   f"  {b}: {v[0]:.1f} GB" for b, v in sorted(sizes.items())))
    return findings


def handler(event, context=None):
    event = event or {}
    overrides = {"dry_run": event["dry_run"]} if "dry_run" in event else {}
    cfg = Config.from_env(overrides)
    now = dt.datetime.now(dt.timezone.utc)
    mode = event.get("mode") or ("jobs" if event.get("source") in ("aws.sagemaker", "aws.batch") else "periodic")
    c = _clients()
    out: dict = {"mode": mode, "dry_run": cfg.dry_run}
    if mode == "periodic":
        out["ec2"] = run_ec2_rules(c, cfg, now)
    if mode in ("periodic", "jobs"):
        out["jobs"] = run_job_rules(c, cfg, now)
    if mode == "hygiene":
        out["hygiene"] = run_hygiene(c, cfg, now)
    logger.info("result %s", json.dumps(out, default=str))
    return out
