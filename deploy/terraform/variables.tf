# Input Variables for Free-Tier Production Cloud Deployment
# All sensitive credentials and API keys are injected via GitHub Secrets / TF_VAR_*

variable "gcp_project_id" {
  description = "Google Cloud Project ID with billing enabled for free-tier activation."
  type        = string
  default     = "hscm-delay-prediction"
}

variable "gcp_region" {
  description = "Google Cloud region supporting Free Tier Cloud Run."
  type        = string
  default     = "us-central1"
}

variable "container_image_uri" {
  description = "GHCR container image URI built by GitHub Actions."
  type        = string
  default     = "ghcr.io/username/hscm-prediction-api:latest"
}

variable "cloudflare_account_id" {
  description = "Cloudflare Account ID for R2 Object Storage."
  type        = string
  default     = "dummy-cloudflare-account-id"
}

variable "r2_bucket_name" {
  description = "Cloudflare R2 Bucket name for model weights and preprocessors."
  type        = string
  default     = "hscm-model-artifacts"
}

variable "r2_access_key_id" {
  description = "Cloudflare R2 S3-compatible API token Access Key ID."
  type        = string
  sensitive   = true
  default     = "mock-r2-key"
}

variable "r2_secret_access_key" {
  description = "Cloudflare R2 S3-compatible API token Secret Access Key."
  type        = string
  sensitive   = true
  default     = "mock-r2-secret"
}

variable "upstash_redis_url" {
  description = "Upstash Serverless Redis connection URL for L2 caching (rediss://...)."
  type        = string
  sensitive   = true
  default     = "rediss://default:dummy@us1-quick-example.upstash.io:6379"
}

variable "neon_database_url" {
  description = "Neon Serverless Postgres connection string for audit log storage."
  type        = string
  sensitive   = true
  default     = "postgresql://user:password@ep-example.us-east-2.aws.neon.tech/neondb?sslmode=require"
}

variable "dagshub_mlflow_uri" {
  description = "DagsHub hosted remote MLflow tracking URI."
  type        = string
  default     = "https://dagshub.com/username/hscm-supply-chain.mlflow"
}

variable "billing_account_id" {
  description = "GCP Billing Account ID for budget alerts (000000-000000-000000)."
  type        = string
  default     = ""
}
