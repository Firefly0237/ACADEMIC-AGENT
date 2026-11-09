output "workload_contract" {
  description = "Non-secret endpoints, instance names and release references for operations."
  value = {
    schema_version = "citequest-gcp-workloads/1"
    https = {
      address = google_compute_global_address.web.address
      domains = var.domains
      urls    = [for domain in var.domains : "https://${domain}"]
    }
    web = {
      regional_mig         = google_compute_region_instance_group_manager.web.id
      zones                = var.zones
      replicas             = var.web_replicas
      health_path          = "/healthz"
      readiness_path       = "/readyz"
      max_unavailable      = 0
      max_surge            = length(var.zones)
      draining_seconds     = 300
    }
    workers = { for name, instance in google_compute_instance.worker : name => {
      instance    = instance.name
      zone        = instance.zone
      private_ip  = instance.network_interface[0].network_ip
      state_disk  = google_compute_disk.state[name].id
      state_mount = "/var/lib/citequest/state"
      activation  = name == "parser" ? "fixed one-File trusted broker invocation" : "drain, reconcile, controlled reboot"
    } }
    parser = {
      image           = var.roles.parser.image
      host_identity   = google_service_account.parser.email
      child_network   = "none"
      child_secrets   = false
      child_docker    = false
      entrypoint      = "python -S scripts/s04_linux/worker_entry.py"
      dispatch        = "scripts/s04_linux/private_host_broker.py"
      dispatch_policy = "one File and immutable execution profile; no automatic exchange replay"
    }
    external_services = {
      postgres_connection_name = var.foundation.postgres.connection_name
      business_bucket          = var.foundation.blobs.bucket_name
      source_cas               = var.source_cas_nfs
      index                    = "private Milvus/TEI endpoints in checksummed search.json"
    }
    releases = { for name, role in var.roles : name => {
      image = role.image
      files = { for path, file in role.config_files : path => {
        uri    = "gs://${var.config_bucket}/${file.object}#${file.generation}"
        sha256 = file.sha256
      } }
    } }
  }
}
