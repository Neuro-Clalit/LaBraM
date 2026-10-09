#!/usr/bin/env bash
# Package and deploy the cost-guard stack (see README.md).
#   ./deploy.sh                       # dry-run mode: emails what it *would* stop
#   DRY_RUN=false ./deploy.sh         # enforce
#   ./deploy.sh MaxRuntimeHours=96    # any extra template parameter overrides
set -euo pipefail
cd "$(dirname "$0")"
: "${ALERT_EMAIL:?set ALERT_EMAIL}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${ARTIFACT_BUCKET:-sagemaker-${REGION}-${ACCOUNT}}"
rm -rf src/__pycache__
aws cloudformation package --region "$REGION" --template-file template.yaml \
  --s3-bucket "$BUCKET" --s3-prefix cost-guard --output-template-file /tmp/cost-guard.packaged.yaml
aws cloudformation deploy --region "$REGION" --stack-name cost-guard \
  --template-file /tmp/cost-guard.packaged.yaml --capabilities CAPABILITY_NAMED_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides AlertEmail="$ALERT_EMAIL" DryRun="${DRY_RUN:-true}" "$@"
aws cloudformation describe-stacks --region "$REGION" --stack-name cost-guard \
  --query 'Stacks[0].Outputs' --output table
