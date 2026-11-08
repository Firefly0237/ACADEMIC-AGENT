output "foundation_contract" {
  description = "Non-secret deployment handoff only; unknown outputs do not imply any resource has been created."
  value = {
    schema_version = "citequest-gcp-foundation/1"
    project_id     = var.project_id
    region         = var.region
    network = {
      id          = google_compute_network.product.id
      self_link   = google_compute_network.product.self_link
      subnet_id   = google_compute_subnetwork.services.id
      subnet_name = google_compute_subnetwork.services.name
      subnet_cidr = var.subnet_cidr
    }
    service_accounts = { for kind, account in google_service_account.runtime : kind => account.email }
    blobs = {
      bucket_name = google_storage_bucket.blobs.name
      bucket_uri  = "gs://${google_storage_bucket.blobs.name}"
      backend     = "gcs"
    }
    postgres = {
      database_version = "POSTGRES_17"
      connection_name  = google_sql_database_instance.product.connection_name
      private_ip       = google_sql_database_instance.product.private_ip_address
      database         = google_sql_database.application.name
      iam_usernames    = { for kind, user in google_sql_user.runtime : kind => user.name }
      tls_required     = true
      iam_auth         = true
      availability     = var.sql_availability_type
    }
    secret_refs = {
      api    = { for name, ref in var.api_secret_refs : name => "projects/${var.project_id}/secrets/${ref.secret_id}/versions/${ref.version}" }
      worker = { for name, ref in var.worker_secret_refs : name => "projects/${var.project_id}/secrets/${ref.secret_id}/versions/${ref.version}" }
    }
    pending_modules = [
      "Reviewed Cloud Run production image + OIDC/session + GCS business bridge + VPC/IAM connector wiring",
      "Reviewed private GCE model/index and worker images, disks, network/egress policy and restricted Linux parser",
      "SQL role/schema/privilege bootstrap, migrations and restored-data verification",
      "Remote state governance, real IAM/API/region/HA/backup/restore/cost/capacity acceptance"
    ]
  }
}
