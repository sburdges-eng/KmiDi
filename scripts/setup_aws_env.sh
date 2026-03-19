#!/usr/bin/env bash
# setup_aws_env.sh - Verify and finalize AWS environment configuration

set -e

ENV_FILE="KmiDi/.env.aws"

echo "===================================================="
echo " KmiDi AWS Environment Setup"
echo "===================================================="

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found."
    exit 1
fi

source "$ENV_FILE"

echo "Using Profile: $AWS_PROFILE"
echo "Region: $AWS_DEFAULT_REGION"
echo "Account: $AWS_ACCOUNT_ID"
echo ""

# Check if AWS CLI can communicate
echo "Checking AWS connectivity..."
if aws sts get-caller-identity --profile "$AWS_PROFILE" > /dev/null 2>&1; then
    echo "✅ AWS CLI is authorized for profile: $AWS_PROFILE"
else
    echo "❌ AWS CLI authorization failed for profile: $AWS_PROFILE"
    echo "Please run: aws sso login --profile $AWS_PROFILE"
    echo "Or verify your access keys in ~/.aws/credentials"
fi

echo ""
echo "Next Steps:"
echo "1. Edit $ENV_FILE to fill in your KMIDI_AWS_AMI_ID and SAGEMAKER_EXECUTION_ROLE_ARN."
echo "2. Ensure your IAM role has permissions for S3, SageMaker, and Lambda."
echo "3. Run 'source $ENV_FILE' before launching cloud training scripts."
echo "===================================================="
