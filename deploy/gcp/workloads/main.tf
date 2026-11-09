locals {
  labels        = merge(var.labels, { app = "citequest", component = "workloads" })
  worker_roles  = { for name, role in var.roles : name => role if name != "web" }
  source_volume = "cq-source-cas-${substr(sha256(jsonencode(var.source_cas_nfs)), 0, 12)}"
  accounts = {
    web     = var.foundation.service_accounts.api
    agent   = var.foundation.service_accounts.worker
    updater = var.foundation.service_accounts.worker
    export  = var.foundation.service_accounts.worker
    parser  = google_service_account.parser.email
  }
  bootstrap_accounts = {
    api    = var.foundation.service_accounts.api
    worker = var.foundation.service_accounts.worker
    parser = google_service_account.parser.email
  }
  registry_grants = { for grant in flatten([
    for repo in var.artifact_registry_repositories : [
      for kind, account in local.bootstrap_accounts : {
        key     = "${repo.project}/${repo.location}/${repo.repository}/${kind}"
        repo    = repo
        account = account
      }
    ]
  ]) : grant.key => grant }
}

# The parser host can pull releases, but has no business bucket, SQL or secret IAM.
# Each untrusted parser child additionally runs with --network=none.
resource "google_service_account" "parser" {
  project      = var.foundation.project_id
  account_id   = "${var.name_prefix}-parser"
  display_name = "CiteQuest secretless parser job host"
}

resource "google_storage_bucket_iam_member" "config_read" {
  for_each = local.bootstrap_accounts
  bucket   = var.config_bucket
  role     = "roles/storage.objectViewer"
  member   = "serviceAccount:${each.value}"
}

resource "google_artifact_registry_repository_iam_member" "images" {
  for_each   = local.registry_grants
  project    = each.value.repo.project
  location   = each.value.repo.location
  repository = each.value.repo.repository
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${each.value.account}"
}

resource "google_compute_router" "egress" {
  project = var.foundation.project_id
  name    = "${var.name_prefix}-egress"
  region  = var.foundation.region
  network = var.foundation.network.id
}

resource "google_compute_router_nat" "egress" {
  project                            = var.foundation.project_id
  name                               = "${var.name_prefix}-egress"
  router                             = google_compute_router.egress.name
  region                             = var.foundation.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"
  min_ports_per_vm                    = 128
  subnetwork {
    name                    = var.foundation.network.subnet_id
    source_ip_ranges_to_nat = ["ALL_IP_RANGES"]
  }
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Only Google's global external Application LB proxies/health checks reach Web.
resource "google_compute_firewall" "web_lb" {
  project       = var.foundation.project_id
  name          = "${var.name_prefix}-web-lb"
  network       = var.foundation.network.id
  direction     = "INGRESS"
  source_ranges = ["35.191.0.0/16", "130.211.0.0/22"]
  target_tags   = ["${var.name_prefix}-web"]
  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }
}

resource "google_compute_instance_template" "web" {
  project      = var.foundation.project_id
  name_prefix  = "${var.name_prefix}-web-"
  machine_type = var.roles.web.machine_type
  region       = var.foundation.region
  tags         = ["${var.name_prefix}-web"]
  labels       = merge(local.labels, { role = "web" })
  disk {
    source_image = var.cos_image
    auto_delete  = true
    boot         = true
    disk_size_gb = 30
    disk_type    = "pd-balanced"
  }
  network_interface {
    subnetwork = var.foundation.network.subnet_id
    # No access_config: all VMs have private addresses only.
  }
  service_account {
    email  = local.accounts.web
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
  metadata = {
    "user-data"              = local.cloud_config.web
    "enable-oslogin"         = "TRUE"
    "block-project-ssh-keys" = "TRUE"
    "disable-legacy-endpoints" = "TRUE"
  }
  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }
  lifecycle {
    create_before_destroy = true
  }
}

resource "google_compute_health_check" "web_live" {
  project             = var.foundation.project_id
  name                = "${var.name_prefix}-web-live"
  check_interval_sec  = 10
  timeout_sec         = 5
  healthy_threshold   = 2
  unhealthy_threshold = 6
  http_health_check {
    port         = 8080
    request_path = "/healthz"
  }
}

resource "google_compute_health_check" "web_ready" {
  project             = var.foundation.project_id
  name                = "${var.name_prefix}-web-ready"
  check_interval_sec  = 10
  timeout_sec         = 5
  healthy_threshold   = 2
  unhealthy_threshold = 3
  http_health_check {
    port         = 8080
    request_path = "/readyz"
  }
}

resource "google_compute_region_instance_group_manager" "web" {
  project                   = var.foundation.project_id
  name                      = "${var.name_prefix}-web"
  base_instance_name        = "${var.name_prefix}-web"
  region                    = var.foundation.region
  distribution_policy_zones = var.zones
  target_size               = var.web_replicas
  version {
    name              = "release-${substr(sha256(local.cloud_config.web), 0, 12)}"
    instance_template = google_compute_instance_template.web.self_link_unique
  }
  named_port {
    name = "http"
    port = 8080
  }
  auto_healing_policies {
    health_check      = google_compute_health_check.web_live.id
    initial_delay_sec = 300
  }
  update_policy {
    type                         = "PROACTIVE"
    minimal_action               = "REPLACE"
    most_disruptive_allowed_action = "REPLACE"
    replacement_method           = "SUBSTITUTE"
    max_unavailable_fixed        = 0
    # For a regional MIG, a nonzero fixed surge must be >= its zone count.
    max_surge_fixed              = length(var.zones)
    min_ready_sec                = 60
  }
  depends_on = [google_compute_router_nat.egress, google_storage_bucket_iam_member.config_read, google_artifact_registry_repository_iam_member.images]
}

resource "google_compute_disk" "state" {
  for_each = local.worker_roles
  project  = var.foundation.project_id
  name     = "${var.name_prefix}-${each.key}-state"
  zone     = each.value.zone
  type     = "pd-balanced"
  size     = each.value.state_disk_gb
  labels   = merge(local.labels, { role = each.key })
  lifecycle {
    prevent_destroy = true
  }
}

# Stateful/queue consumers deliberately do not use a proactive MIG rollout.
# A metadata change stages a release; only a completed explicit drain/reboot
# activates it. prevent_destroy blocks image/machine/zone replacement via apply.
resource "google_compute_instance" "worker" {
  for_each                  = local.worker_roles
  project                   = var.foundation.project_id
  name                      = "${var.name_prefix}-${each.key}"
  zone                      = each.value.zone
  machine_type              = each.value.machine_type
  allow_stopping_for_update = false
  deletion_protection       = true
  labels                    = merge(local.labels, { role = each.key })
  tags                      = ["${var.name_prefix}-${each.key}"]
  boot_disk {
    auto_delete = true
    initialize_params {
      image = var.cos_image
      size  = each.key == "parser" ? 100 : 40
      type  = "pd-balanced"
    }
  }
  attached_disk {
    source      = google_compute_disk.state[each.key].id
    device_name = "citequest-state"
    mode        = "READ_WRITE"
  }
  network_interface {
    subnetwork = var.foundation.network.subnet_id
  }
  service_account {
    email  = local.accounts[each.key]
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
  metadata = {
    "user-data"                = local.cloud_config[each.key]
    "enable-oslogin"           = "TRUE"
    "block-project-ssh-keys"   = "TRUE"
    "disable-legacy-endpoints" = "TRUE"
  }
  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
    provisioning_model  = "STANDARD"
  }
  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }
  lifecycle {
    prevent_destroy = true
  }
  depends_on = [google_compute_router_nat.egress, google_storage_bucket_iam_member.config_read, google_artifact_registry_repository_iam_member.images]
}

resource "google_compute_global_address" "web" {
  project = var.foundation.project_id
  name    = "${var.name_prefix}-web"
}

resource "google_compute_backend_service" "web" {
  project                         = var.foundation.project_id
  name                            = "${var.name_prefix}-web"
  protocol                        = "HTTP"
  port_name                       = "http"
  load_balancing_scheme           = "EXTERNAL_MANAGED"
  timeout_sec                     = 600
  connection_draining_timeout_sec = 300
  health_checks                   = [google_compute_health_check.web_ready.id]
  backend {
    group           = google_compute_region_instance_group_manager.web.instance_group
    balancing_mode  = "UTILIZATION"
    max_utilization = 0.8
  }
}

resource "google_compute_url_map" "web" {
  project         = var.foundation.project_id
  name            = "${var.name_prefix}-web"
  default_service = google_compute_backend_service.web.id
}

resource "google_compute_managed_ssl_certificate" "web" {
  project = var.foundation.project_id
  name    = "${var.name_prefix}-web"
  managed {
    domains = var.domains
  }
}

resource "google_compute_ssl_policy" "web" {
  project         = var.foundation.project_id
  name            = "${var.name_prefix}-web"
  profile         = "MODERN"
  min_tls_version = "TLS_1_2"
}

resource "google_compute_target_https_proxy" "web" {
  project          = var.foundation.project_id
  name             = "${var.name_prefix}-web"
  url_map          = google_compute_url_map.web.id
  ssl_certificates = [google_compute_managed_ssl_certificate.web.id]
  ssl_policy       = google_compute_ssl_policy.web.id
}

resource "google_compute_global_forwarding_rule" "https" {
  project               = var.foundation.project_id
  name                  = "${var.name_prefix}-https"
  ip_address            = google_compute_global_address.web.address
  target                = google_compute_target_https_proxy.web.id
  port_range            = "443"
  load_balancing_scheme = "EXTERNAL_MANAGED"
}

resource "google_compute_url_map" "https_redirect" {
  project = var.foundation.project_id
  name    = "${var.name_prefix}-https-redirect"
  default_url_redirect {
    https_redirect         = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query            = false
  }
}

resource "google_compute_target_http_proxy" "https_redirect" {
  project = var.foundation.project_id
  name    = "${var.name_prefix}-https-redirect"
  url_map = google_compute_url_map.https_redirect.id
}

resource "google_compute_global_forwarding_rule" "http_redirect" {
  project               = var.foundation.project_id
  name                  = "${var.name_prefix}-http-redirect"
  ip_address            = google_compute_global_address.web.address
  target                = google_compute_target_http_proxy.https_redirect.id
  port_range            = "80"
  load_balancing_scheme = "EXTERNAL_MANAGED"
}
