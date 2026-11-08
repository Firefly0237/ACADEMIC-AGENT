locals {
  labels = merge(var.labels, { app = "citequest", foundation = "cf02" })
  runtime_accounts = {
    api    = "Web/API runtime; application ACL/OIDC is not supplied by this identity."
    worker = "Trusted Agent/ETL executor; never attach this identity to Docling containers."
    index  = "Model/index VM identity; no business bucket or Cloud SQL permissions."
  }
  sql_clients = toset(["api", "worker"])
}

resource "google_compute_network" "product" {
  project                 = var.project_id
  name                    = "${var.name_prefix}-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
  description             = "CiteQuest private service plane; no workloads or ingress firewall in CF-02."

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_compute_subnetwork" "services" {
  project                  = var.project_id
  name                     = "${var.name_prefix}-services"
  region                   = var.region
  network                  = google_compute_network.product.id
  ip_cidr_range            = var.subnet_cidr
  private_ip_google_access = true
}

resource "google_compute_global_address" "private_services" {
  project       = var.project_id
  name          = "${var.name_prefix}-private-services"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  address       = cidrhost(var.private_services_cidr, 0)
  prefix_length = tonumber(split("/", var.private_services_cidr)[1])
  network       = google_compute_network.product.id
}

resource "google_service_networking_connection" "private_services" {
  network                 = google_compute_network.product.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_services.name]
}

resource "google_service_account" "runtime" {
  for_each     = local.runtime_accounts
  project      = var.project_id
  account_id   = "${var.name_prefix}-${each.key}"
  display_name = "CiteQuest ${each.key} runtime"
  description  = each.value
}

# Additive member resources do not replace the project's complete IAM policy.
# These two Cloud SQL roles are project-scoped; SQL privileges are NOT granted.
resource "google_project_iam_member" "sql_connect" {
  for_each = local.sql_clients
  project  = var.project_id
  role     = "roles/cloudsql.client"
  member   = "serviceAccount:${google_service_account.runtime[each.key].email}"
}

resource "google_project_iam_member" "sql_login" {
  for_each = local.sql_clients
  project  = var.project_id
  role     = "roles/cloudsql.instanceUser"
  member   = "serviceAccount:${google_service_account.runtime[each.key].email}"
}

resource "google_storage_bucket" "blobs" {
  project                     = var.project_id
  name                        = var.blob_bucket_name
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  deletion_policy             = "PREVENT"
  labels                      = local.labels

  versioning {
    enabled = true
  }

  soft_delete_policy {
    retention_duration_seconds = 604800
  }

  # No blind age-based GC: active PG/generation references must govern deletion.
  lifecycle {
    prevent_destroy = true
  }
}

# API writes immutable new objects and reads; trusted worker alone handles purge.
resource "google_storage_bucket_iam_member" "api_create" {
  bucket = google_storage_bucket.blobs.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.runtime["api"].email}"
}

resource "google_storage_bucket_iam_member" "api_read" {
  bucket = google_storage_bucket.blobs.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.runtime["api"].email}"
}

resource "google_storage_bucket_iam_member" "worker_objects" {
  bucket = google_storage_bucket.blobs.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.runtime["worker"].email}"
}

resource "google_secret_manager_secret_iam_member" "api" {
  for_each  = toset([for ref in values(var.api_secret_refs) : ref.secret_id])
  project   = var.project_id
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime["api"].email}"
}

resource "google_secret_manager_secret_iam_member" "worker" {
  for_each  = toset([for ref in values(var.worker_secret_refs) : ref.secret_id])
  project   = var.project_id
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime["worker"].email}"
}

resource "google_sql_database_instance" "product" {
  project             = var.project_id
  name                = var.sql_instance_name
  region              = var.region
  database_version    = "POSTGRES_17"
  deletion_protection = true

  settings {
    edition                     = "ENTERPRISE"
    tier                        = var.sql_tier
    availability_type           = var.sql_availability_type
    disk_type                   = "PD_SSD"
    disk_size                   = var.sql_disk_size_gb
    disk_autoresize             = true
    disk_autoresize_limit       = var.sql_disk_limit_gb
    deletion_protection_enabled = true
    user_labels                 = local.labels

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.product.id
      ssl_mode        = "ENCRYPTED_ONLY"
    }

    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "on"
    }

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 7
        retention_unit   = "COUNT"
      }
    }

    maintenance_window {
      day          = 7
      hour         = 4
      update_track = "stable"
    }
  }

  depends_on = [google_service_networking_connection.private_services]

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_sql_database" "application" {
  project         = var.project_id
  name            = var.database_name
  instance        = google_sql_database_instance.product.name
  deletion_policy = "PREVENT"

  lifecycle {
    prevent_destroy = true
  }
}

# No password values in configuration/state. IAM identities alone do not grant
# application schema access; controlled SQL bootstrap/migrations are still needed.
resource "google_sql_user" "runtime" {
  for_each = local.sql_clients
  project  = var.project_id
  name     = trimsuffix(google_service_account.runtime[each.key].email, ".gserviceaccount.com")
  instance = google_sql_database_instance.product.name
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
}
