"""ECS Fargate entrypoint — calls the same logic as the Lambda handler."""
import sys
from handler import lambda_handler

result = lambda_handler({}, None)
print(result)
sys.exit(0 if result.get("statusCode") == 200 else 1)
