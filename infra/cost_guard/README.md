# cost-guard

Account-level guard against forgotten EC2/SageMaker spend. One CloudFormation stack
(`template.yaml`) with one Lambda (`src/cost_guard.py`), an SNS email topic, budgets, and
Cost Anomaly Detection.

## Rules

| # | Trigger | Action |
|---|---------|--------|
| 1 | EC2 running ≥ 4 h, **every** hour of the last 4 h averaged CPU < 1 %, and GPU utilisation (if the CloudWatch agent reports it) never ≥ 5 % | stop + email |
| 2 | EC2 running ≥ 36 h | email (once per start) |
| 3 | EC2 running ≥ 72 h | warning email at 66 h, stop + email at 72 h |
| 4 | > 16 active GPU jobs (SageMaker training/processing on `ml.g*`/`ml.p*`, Batch jobs requesting GPUs) | email (repeats at most every 6 h, sooner if the count grows 50 %) |
| 5 | > 128 active jobs (same sources, any instance type) | newest jobs beyond 128 are stopped + email |
| 6 | daily 06:00 UTC | email listing leaks: unattached EBS, free Elastic IPs, NAT gateways, instances stopped > 14 days, non-DLM snapshots > 30 days, SageMaker endpoints/notebooks/Studio apps/warm pools in service, S3 total > 400 GB or a bucket growing > 25 GB/day, buckets without an incomplete-multipart-upload cleanup rule |
| — | monthly cost > 50/80/100 % of $70, or forecast > $70; daily cost > $40; cost anomaly ≥ $10 | AWS Budgets / Cost Anomaly Detection email |
| — | the Lambda itself errors | CloudWatch alarm email |

Thresholds are stack parameters (see `template.yaml`); "running" is time since the last
start (`LaunchTime`). The 1 % CPU threshold was calibrated on this account's g5.2xlarge
history: idle hours sit at ~0.1 %, an interactive shell/IDE session at 0.6–1.6 %, training
at 10–25 % (a GPU job keeps ≥ 1 core busy = 12.5 % on 8 vCPUs).

**Exemption:** tag an instance `CostGuardSkipUntil=<ISO date>` (e.g. `2026-10-10T00:00Z`) to
suspend rules 1 and 3 until then; emails still arrive. Spot, ASG, Batch, and EKS instances are
never stopped (their owning service would fight it); they only get emails.

Stopping wipes instance-store disks (e.g. `/opt/dlami/nvme`); EBS volumes are kept.

## Deploy

```bash
ALERT_EMAIL=you@example.com ./deploy.sh            # dry run: emails what it would do
ALERT_EMAIL=you@example.com DRY_RUN=false ./deploy.sh   # enforce
ALERT_EMAIL=you@example.com ./deploy.sh MaxRuntimeHours=96 MonthlyBudgetUsd=100
```

Confirm the SNS subscription email after the first deploy, or nothing is delivered.

Test an invocation without side effects:

```bash
aws lambda invoke --function-name cost-guard \
  --payload '{"mode":"periodic","dry_run":true}' --cli-binary-format raw-in-base64-out /dev/stdout
```

### GPU metrics (rule 1)

Without the CloudWatch agent the idle rule is CPU-only. On a GPU box:

```bash
aws ec2 associate-iam-instance-profile --instance-id <id> \
  --iam-instance-profile Name=cost-guard-ec2-metrics
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 \
  -c file:$(pwd)/cloudwatch-agent-gpu.json -s
```

This publishes `CWAgent/nvidia_smi_utilization_gpu` per instance (~$0.60/month).

## Remove

```bash
aws cloudformation delete-stack --stack-name cost-guard
```
