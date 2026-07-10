#!/usr/bin/env bash
# deploy.sh — deploy or tear down the AMFI NAV ingestion pipeline stack.
#
# Usage:
#   ./infra/deploy.sh deploy  --bucket my-nav-bucket [--stack amfi-pipeline] [--region ap-south-1]
#   ./infra/deploy.sh destroy [--stack amfi-pipeline] [--region ap-south-1]
#   ./infra/deploy.sh invoke  [--stack amfi-pipeline] [--region ap-south-1]
#   ./infra/deploy.sh status  [--stack amfi-pipeline] [--region ap-south-1]
#
# Prerequisites:
#   aws CLI configured with credentials that can create IAM/Lambda/S3/Scheduler resources.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/cloudformation/amfi-pipeline.yaml"

# ── Defaults ────────────────────────────────────────────────────────────────
STACK_NAME="amfi-pipeline"
REGION="${AWS_REGION:-ap-south-1}"
S3_BUCKET="mf-finance-kb"
S3_PREFIX="bronze/amfi/nav"
SCHEDULE="cron(30 3 * * ? *)"   # 09:00 IST daily

# ── Argument parsing ─────────────────────────────────────────────────────────
COMMAND="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack)    STACK_NAME="$2";  shift 2 ;;
    --bucket)   S3_BUCKET="$2";   shift 2 ;;
    --prefix)   S3_PREFIX="$2";   shift 2 ;;
    --region)   REGION="$2";      shift 2 ;;
    --schedule) SCHEDULE="$2";    shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
_aws() { aws --region "$REGION" "$@"; }

_lambda_name() {
  _aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='LambdaFunctionName'].OutputValue" \
    --output text 2>/dev/null
}

# ── Commands ─────────────────────────────────────────────────────────────────
cmd_deploy() {
  echo "▶ Deploying stack '${STACK_NAME}' in ${REGION}…"
  echo "  Bucket  : ${S3_BUCKET}"
  echo "  Prefix  : ${S3_PREFIX}"
  echo "  Schedule: ${SCHEDULE}"
  echo

  _aws cloudformation deploy \
    --stack-name    "$STACK_NAME" \
    --template-file "$TEMPLATE" \
    --capabilities  CAPABILITY_NAMED_IAM \
    --region        "$REGION" \
    --parameter-overrides \
        S3BucketName="$S3_BUCKET" \
        S3Prefix="$S3_PREFIX" \
        ScheduleExpression="$SCHEDULE" \
    --tags \
        Project=financial-pipeline \
        Component=amfi-nav-ingestion

  echo
  echo "✓ Stack deployed. Outputs:"
  _aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs" \
    --output table
}

cmd_destroy() {
  echo "▶ Deleting stack '${STACK_NAME}'…"
  echo "  NOTE: The S3 bucket has DeletionPolicy=Retain — data is preserved."
  read -r -p "  Type 'yes' to confirm: " confirm
  [[ "$confirm" == "yes" ]] || { echo "Aborted."; exit 0; }

  _aws cloudformation delete-stack --stack-name "$STACK_NAME"
  echo "  Waiting for deletion to complete…"
  _aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME"
  echo "✓ Stack deleted."
}

cmd_invoke() {
  FUNC=$(_lambda_name)
  if [[ -z "$FUNC" ]]; then
    echo "ERROR: could not find Lambda function in stack '${STACK_NAME}'"; exit 1
  fi
  echo "▶ Invoking Lambda: ${FUNC}"
  _aws lambda invoke \
    --function-name "$FUNC" \
    --payload '{}' \
    --cli-binary-format raw-in-base64-out \
    /dev/stdout
}

cmd_status() {
  echo "▶ Stack status:"
  _aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].{Status:StackStatus,Updated:LastUpdatedTime}" \
    --output table

  echo
  echo "▶ Outputs:"
  _aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs" \
    --output table
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
case "$COMMAND" in
  deploy)  cmd_deploy  ;;
  destroy) cmd_destroy ;;
  invoke)  cmd_invoke  ;;
  status)  cmd_status  ;;
  *)
    echo "Usage: $0 {deploy|destroy|invoke|status} [options]"
    echo "  deploy  --bucket <name>  [--stack <name>] [--region <r>] [--prefix <p>] [--schedule <cron>]"
    echo "  destroy [--stack <name>] [--region <r>]"
    echo "  invoke  [--stack <name>] [--region <r>]"
    echo "  status  [--stack <name>] [--region <r>]"
    exit 1
    ;;
esac
