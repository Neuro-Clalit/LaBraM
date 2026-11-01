#!/usr/bin/env bash
# Deploy the private HEEDB bucket + KMS key + reader policy.
#   ./deploy.sh                                  # defaults (heedb-private-<account>)
#   ./deploy.sh ReaderRoles=labram-dataprep-job  # template parameter overrides
#
# Uses the ambient AWS credentials; CloudFormation needs an MFA session on this
# account (the default profile is MFA-backed, see github.com/gugas81/common).
# The bucket is retained when the stack is deleted (DeletionPolicy: Retain).
set -euo pipefail
cd "$(dirname "$0")"
REGION="${AWS_REGION:-us-east-1}"
aws cloudformation deploy --region "$REGION" --stack-name heedb-storage \
  --template-file template.yaml --capabilities CAPABILITY_NAMED_IAM --no-fail-on-empty-changeset \
  --parameter-overrides "$@"
aws cloudformation describe-stacks --region "$REGION" --stack-name heedb-storage \
  --query 'Stacks[0].Outputs' --output table
