#!/usr/bin/env bash
# deploy_mf_nav_sync.sh — deploy/manage the mf-nav-sync (mfapi.in) stack.
#
# Usage:
#   ./infra/deploy_mf_nav_sync.sh deploy   [--stack mf-nav-sync] [--region us-east-1] [--image <ecr-uri>]
#   ./infra/deploy_mf_nav_sync.sh destroy  [--stack mf-nav-sync] [--region us-east-1]
#   ./infra/deploy_mf_nav_sync.sh status   [--stack mf-nav-sync] [--region us-east-1]
#   ./infra/deploy_mf_nav_sync.sh backfill [--stack mf-nav-sync] [--region us-east-1] [--start-date 2000-01-01]
#       Runs a ONE-OFF full historical sync via `ecs run-task` with a command
#       override (the deployed task definition's default command only does a
#       short incremental sync — see infra/cloudformation/mf-nav-sync.yaml).
#
# Prerequisites:
#   aws CLI configured with credentials that can create/describe CloudFormation,
#   ECS, IAM, and EventBridge Scheduler resources; POSTGRES_URL available in
#   the environment (e.g. via `set -a && source .env && set +a`) for `deploy`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/cloudformation/mf-nav-sync.yaml"

STACK_NAME="mf-nav-sync"
REGION="${AWS_REGION:-us-east-1}"
IMAGE="468895762981.dkr.ecr.us-east-1.amazonaws.com/amfi-doc-processor:latest"
START_DATE="2000-01-01"

COMMAND="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack)      STACK_NAME="$2"; shift 2 ;;
    --region)     REGION="$2";     shift 2 ;;
    --image)      IMAGE="$2";      shift 2 ;;
    --start-date) START_DATE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

_aws() { aws --region "$REGION" "$@"; }

cmd_deploy() {
  : "${POSTGRES_URL:?POSTGRES_URL must be set in the environment}"
  echo "▶ Deploying stack '${STACK_NAME}' in ${REGION} with image ${IMAGE}…"

  if _aws cloudformation describe-stacks --stack-name "$STACK_NAME" >/dev/null 2>&1; then
    _aws cloudformation update-stack \
      --stack-name "$STACK_NAME" \
      --template-body "file://${TEMPLATE}" \
      --capabilities CAPABILITY_NAMED_IAM \
      --parameters \
          ParameterKey=EcrImage,ParameterValue="$IMAGE" \
          ParameterKey=PostgresUrl,ParameterValue="$POSTGRES_URL"
    _aws cloudformation wait stack-update-complete --stack-name "$STACK_NAME"
  else
    _aws cloudformation create-stack \
      --stack-name "$STACK_NAME" \
      --template-body "file://${TEMPLATE}" \
      --capabilities CAPABILITY_NAMED_IAM \
      --parameters \
          ParameterKey=EcrImage,ParameterValue="$IMAGE" \
          ParameterKey=PostgresUrl,ParameterValue="$POSTGRES_URL" \
      --tags Key=Project,Value=financial-pipeline Key=Component,Value=mf-nav-sync
    _aws cloudformation wait stack-create-complete --stack-name "$STACK_NAME"
  fi

  echo "✓ Stack deployed. Outputs:"
  _aws cloudformation describe-stacks --stack-name "$STACK_NAME" --query "Stacks[0].Outputs" --output table
}

cmd_destroy() {
  echo "▶ Deleting stack '${STACK_NAME}'…"
  read -r -p "  Type 'yes' to confirm: " confirm
  [[ "$confirm" == "yes" ]] || { echo "Aborted."; exit 0; }
  _aws cloudformation delete-stack --stack-name "$STACK_NAME"
  _aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME"
  echo "✓ Stack deleted."
}

cmd_status() {
  _aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].{Status:StackStatus,Updated:LastUpdatedTime}" \
    --output table
  echo
  _aws cloudformation describe-stacks --stack-name "$STACK_NAME" --query "Stacks[0].Outputs" --output table
}

cmd_backfill() {
  TASKDEF=$(_aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='TaskDefinitionArn'].OutputValue" --output text)
  CLUSTER=$(_aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='ClusterName'].OutputValue" --output text)

  CMD=$(python3 - "$START_DATE" <<'PYEOF'
import json, sys
start_date = sys.argv[1]
code = (
    'import sys; sys.path.insert(0, "src")\n'
    'from financial_pipeline.config import settings\n'
    'from financial_pipeline.mf_ingestion.sync import run_sync\n'
    'result = run_sync(\n'
    '    postgres_url=settings.postgres_url,\n'
    '    s3_bucket=settings.s3_bucket,\n'
    '    s3_prefix=settings.mf_scheme_master_s3_prefix,\n'
    '    aws_region=settings.aws_region,\n'
    f'    start_date="{start_date}",\n'
    '    request_delay_seconds=settings.mfapi_request_delay_seconds,\n'
    ')\n'
    'print(result)\n'
)
print(json.dumps(["python", "-c", code]))
PYEOF
)

  echo "▶ Running one-off full backfill (start_date=${START_DATE}) on ${TASKDEF}…"
  _aws ecs run-task \
    --cluster "$CLUSTER" \
    --task-definition "$TASKDEF" \
    --launch-type FARGATE \
    --network-configuration '{
      "awsvpcConfiguration": {
        "subnets":        ["subnet-089bc3cd1c8760332"],
        "securityGroups": ["sg-0eecd839abfcf4daa"],
        "assignPublicIp": "ENABLED"
      }
    }' \
    --overrides "{\"containerOverrides\": [{\"name\": \"mf-nav-sync\", \"command\": ${CMD}}]}" \
    --query "tasks[0].{TaskArn:taskArn,LastStatus:lastStatus}" --output table
}

case "$COMMAND" in
  deploy)   cmd_deploy   ;;
  destroy)  cmd_destroy  ;;
  status)   cmd_status   ;;
  backfill) cmd_backfill ;;
  *)
    echo "Usage: $0 {deploy|destroy|status|backfill} [options]"
    exit 1
    ;;
esac
