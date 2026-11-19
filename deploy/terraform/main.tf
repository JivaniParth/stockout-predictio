# Terraform / OpenTofu Infrastructure Definition
# Production-Ready Free-Tier Architecture for Pharmaceutical Supply Chain Serving
# Compatible with both HashiCorp Terraform >= 1.5.0 and OpenTofu >= 1.6.0

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# =============================================================================
# 1. Google Cloud Run (Serverless Prediction Microservice)
# Free Tier: 2M requests/mo, 360,000 vCPU-sec, 180,000 GiB-sec, 1 GiB egress
# =============================================================================

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

resource "google_cloud_run_v2_service" "hscm_api" {
  name     = "hscm-prediction-service"
  location = var.gcp_region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = 0 # Strict scale-to-zero to avoid idle charges
      max_instance_count = 2 # Hard ceiling to guarantee free-tier compliance
    }

    containers {
      image = var.container_image_uri

      resources {
        limits = {
          cpu    = "1000m" # 1 vCPU
          memory = "512Mi" # 512 MB
        }
      }

      env {
        name  = "PORT"
        value = "8000"
      }
      env {
        name  = "ENVIRONMENT"
        value = "production"
      }
      env {
        name  = "REDIS_URL"
        value = var.upstash_redis_url
      }
      env {
        name  = "DATABASE_URL"
        value = var.neon_database_url
      }
      env {
        name  = "MLFLOW_TRACKING_URI"
        value = var.dagshub_mlflow_uri
      }
      env {
        name  = "R2_BUCKET_NAME"
        value = var.r2_bucket_name
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        period_seconds    = 15
        failure_threshold = 3
      }
    }
  }
}

# Allow public unauthenticated invocations for predictions
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  project  = var.gcp_project_id
  location = google_cloud_run_v2_service.hscm_api.location
  name     = google_cloud_run_v2_service.hscm_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# =============================================================================
# 2. Cloudflare R2 Model Registry & Artifact Store (S3-Compatible)
# Free Tier: 10 GB-month Standard storage, 1M Class A ops, 10M Class B ops, $0 egress
# =============================================================================

provider "aws" {
  alias                       = "cloudflare_r2"
  region                      = "auto"
  access_key                  = var.r2_access_key_id
  secret_key                  = var.r2_secret_access_key
  skip_credentials_validation = true
  skip_region_validation      = true
  skip_requesting_account_id  = true
  skip_metadata_api_check     = true
  s3_use_path_style           = true

  endpoints {
    s3 = "https://${var.cloudflare_account_id}.r2.cloudflarestorage.com"
  }
}

resource "aws_s3_bucket" "model_artifacts" {
  provider = aws.cloudflare_r2
  bucket   = var.r2_bucket_name
}
