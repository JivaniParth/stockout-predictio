# Deployment Outputs

output "cloud_run_service_uri" {
  description = "Public HTTPS URL for Cloud Run API service."
  value       = google_cloud_run_v2_service.hscm_api.uri
}

output "r2_bucket_endpoint" {
  description = "Cloudflare R2 Bucket S3 API Endpoint."
  value       = "https://${var.cloudflare_account_id}.r2.cloudflarestorage.com/${var.r2_bucket_name}"
}

output "free_tier_limits_summary" {
  description = "Reference Free-Tier limits for provisioned resources."
  value = {
    cloud_run       = "2,000,000 requests/mo, 360k vCPU-sec, 180k GiB-sec, 1 GiB egress"
    cloudflare_r2   = "10 GB-month Standard storage, 1M Class A ops, 10M Class B ops, $0 egress fees"
    upstash_redis   = "500,000 commands/month (~16.6k/day), 250 MB storage, 10 GB bandwidth"
    neon_postgres   = "1 GB database storage, 100 CU-hrs/month, scale-to-zero with auto-wake"
    dagshub_mlflow  = "20 GB storage for DVC + MLflow artifacts, free remote tracking"
    github_actions  = "$12/mo compute credit (~2,000 Linux mins), 500 MB GHCR, 10 GB cache"
    huggingface     = "2 vCPU / 16 GB RAM CPU basic, 2 ZeroGPU 48GB VRAM spaces (free accounts)"
    grafana_cloud   = "10,000 metric series, 50 GB logs, 50 GB traces, 14-day retention"
  }
}
