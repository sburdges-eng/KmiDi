#!/usr/bin/env bash
# Setup Google Cloud Project for KmiDi Vertex AI Training

set -e

echo "=== KmiDi Google Cloud Configuration ==="

# 1. Ask for Project ID and Region if not provided
if [ -z "$GCP_PROJECT_ID" ]; then
    read -p "Enter your Google Cloud Project ID: " GCP_PROJECT_ID
fi

if [ -z "$GCP_REGION" ]; then
    read -p "Enter your Google Cloud Region (default: us-central1): " GCP_REGION
    GCP_REGION=${GCP_REGION:-us-central1}
fi

REPO_NAME="kmidi-repo"
BUCKET_NAME="${GCP_PROJECT_ID}-kmidi-staging"
IMAGE_NAME="kmidi-jepa-train"

echo ""
echo "Configuring for Project: $GCP_PROJECT_ID in Region: $GCP_REGION"

# 2. Set gcloud config
gcloud config set project "$GCP_PROJECT_ID"
gcloud config set compute/region "$GCP_REGION"

# 3. Enable necessary APIs
echo "Enabling necessary Google Cloud APIs (Vertex AI, Artifact Registry, Cloud Storage)..."
gcloud services enable \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com

# 4. Create Artifact Registry Repository (if it doesn't exist)
echo "Checking Artifact Registry repository: $REPO_NAME..."
if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$GCP_REGION" >/dev/null 2>&1; then
    echo "Creating repository..."
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$GCP_REGION" \
        --description="Docker repository for KmiDi ML training"
else
    echo "Repository already exists."
fi

# Configure Docker auth
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

# 5. Create GCS Staging Bucket (if it doesn't exist)
echo "Checking GCS Staging Bucket: gs://$BUCKET_NAME..."
if ! gsutil ls "gs://$BUCKET_NAME" >/dev/null 2>&1; then
    echo "Creating staging bucket..."
    gsutil mb -l "$GCP_REGION" "gs://$BUCKET_NAME"
else
    echo "Staging bucket already exists."
fi

# 6. Generate environment variables file
ENV_FILE=".env.gcp"
IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:v1"

cat > "$ENV_FILE" << EOF
# GCP Environment Variables for KmiDi Vertex AI
export GCP_PROJECT_ID="$GCP_PROJECT_ID"
export GCP_REGION="$GCP_REGION"
export GCP_STAGING_BUCKET="gs://$BUCKET_NAME"
export KMIDI_VERTEX_IMAGE_URI="$IMAGE_URI"
EOF

echo ""
echo "=== Configuration Complete ==="
echo "Created $ENV_FILE with your environment variables."
echo ""
echo "To build and push your Docker container, run:"
echo "  docker build -f Dockerfile.vertex -t \$KMIDI_VERTEX_IMAGE_URI ."
echo "  docker push \$KMIDI_VERTEX_IMAGE_URI"
echo ""
echo "Before launching a job, load your environment variables:"
echo "  source $ENV_FILE"
echo ""
