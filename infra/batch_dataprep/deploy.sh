#!/usr/bin/env bash
# Deploy the batch-dataprep stack, then build and push the data-prep image.
#   ./deploy.sh                      # stack + image (subnets of this instance's VPC)
#   ./deploy.sh MaxVcpus=128         # template parameter overrides
#   IMAGE_ONLY=1 ./deploy.sh         # rebuild/push the image only (code changed)
#
# Uses the ambient AWS credentials. The account's MFA-enforcement policy denies
# CloudFormation (and the IAM / ECR / EC2 calls it makes for you) without MFA, so
# run it with an MFA session -- machine-level tooling from the common repo
# (github.com/gugas81/common, machine/bin/aws-mfa-session):
#   aws-mfa-session && AWS_PROFILE=mfa ./deploy.sh
# The generic version of this stack is common/infra/aws_batch_spot.
set -euo pipefail
cd "$(dirname "$0")"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REPO="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/labram-dataprep"

if [[ -z "${IMAGE_ONLY:-}" ]]; then
  if [[ -z "${VPC_ID:-}" ]]; then        # default: the VPC this instance runs in
    TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
    MAC=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/mac)
    VPC_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
      "http://169.254.169.254/latest/meta-data/network/interfaces/macs/$MAC/vpc-id")
  fi
  SUBNETS="${SUBNETS:-$(aws ec2 describe-subnets --region "$REGION" --filters Name=vpc-id,Values="$VPC_ID" \
    --query 'Subnets[?MapPublicIpOnLaunch].SubnetId' --output text | tr '\t' ',')}"
  aws cloudformation deploy --region "$REGION" --stack-name batch-dataprep \
    --template-file template.yaml --capabilities CAPABILITY_NAMED_IAM --no-fail-on-empty-changeset \
    --parameter-overrides VpcId="$VPC_ID" SubnetIds="$SUBNETS" "$@"
fi

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO%%/*}"
TAG="$(git -C ../.. rev-parse --short HEAD)"
CTX="$(mktemp -d)"; trap 'rm -rf "$CTX"' EXIT
cp Dockerfile requirements.txt ../../dataset_maker/make_TUH_npy.py ../../dataset_maker/make_TUAB_npy.py \
   ../../labram/data/tuh_metadata.py ../../labram/data/line_removal.py "$CTX"/
docker build -t "$REPO:latest" -t "$REPO:$TAG" "$CTX"
docker push "$REPO:$TAG"
docker push "$REPO:latest"
aws cloudformation describe-stacks --region "$REGION" --stack-name batch-dataprep \
  --query 'Stacks[0].Outputs' --output table
