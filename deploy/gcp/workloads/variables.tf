variable "foundation" {
  description = "Non-secret foundation_contract output from the foundation module. No credentials or secret payloads."
  type = object({
    schema_version = string
    project_id     = string
    region         = string
    network = object({
      id          = string
      subnet_id   = string
      subnet_name = string
    })
    service_accounts = map(string)
    blobs            = object({ bucket_name = string })
    postgres         = object({ connection_name = string })
    secret_refs      = object({ api = map(string), worker = map(string) })
  })
  validation {
    condition     = var.foundation.schema_version == "citequest-gcp-foundation/1" && alltrue([for key in ["api", "worker"] : contains(keys(var.foundation.service_accounts), key)])
    error_message = "Supply the citequest-gcp-foundation/1 contract with API and worker runtime identities."
  }
}

variable "name_prefix" {
  type        = string
  description = "Environment prefix, distinct from other workloads in the project."
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,18}[a-z0-9]$", var.name_prefix))
    error_message = "Use a lowercase 4-20 character resource prefix."
  }
}

variable "zones" {
  type        = list(string)
  description = "At least two distinct zones in the foundation region, used by the regional Web MIG."
  validation {
    condition     = length(var.zones) >= 2 && length(distinct(var.zones)) == length(var.zones) && alltrue([for zone in var.zones : startswith(zone, "${var.foundation.region}-")])
    error_message = "Supply at least two different zones in the foundation region."
  }
}

variable "cos_image" {
  type        = string
  description = "Pinned Container-Optimized OS image self-link; never an image family."
  validation {
    condition     = can(regex("^projects/cos-cloud/global/images/cos-[a-z0-9-]+$", var.cos_image))
    error_message = "Use a specific projects/cos-cloud/global/images/cos-... image."
  }
}

variable "bootstrap_image" {
  type        = string
  description = "Official Google Cloud CLI bootstrap image pinned by registry digest."
  validation {
    condition     = can(regex("^gcr\\.io/google\\.com/cloudsdktool/google-cloud-cli(:[A-Za-z0-9._-]+)?@sha256:[a-f0-9]{64}$", var.bootstrap_image))
    error_message = "Supply the official Google Cloud CLI image and its verified sha256 digest."
  }
}

variable "sql_proxy_image" {
  type        = string
  description = "Verified Cloud SQL Auth Proxy 2.27.0 digest; no guessed/default digest."
  validation {
    condition     = can(regex("^gcr\\.io/cloud-sql-connectors/cloud-sql-proxy:2\\.27\\.0@sha256:[a-f0-9]{64}$", var.sql_proxy_image))
    error_message = "Supply gcr.io/cloud-sql-connectors/cloud-sql-proxy:2.27.0@sha256:<verified digest>."
  }
}

variable "config_bucket" {
  type        = string
  description = "Existing private GCS bucket containing immutable non-secret release/config artifacts."
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,220}[a-z0-9]$", var.config_bucket))
    error_message = "Supply a GCS bucket name without gs:// or an object path."
  }
}

variable "roles" {
  description = "Independent role images and non-secret deployment references. Parser is a prepared one-shot job host, not a queue daemon."
  type = map(object({
    image          = string
    machine_type   = string
    memory_mb      = number
    cpus           = number
    state_disk_gb  = optional(number, 30)
    zone           = optional(string)
    command        = optional(list(string), [])
    config_files = map(object({
      object     = string
      generation = string
      sha256     = string
    }))
    secret_files = optional(map(string), {})
    renderer_archive = optional(object({
      config_file = string
      image_id    = string
    }))
  }))
  validation {
    condition     = toset(keys(var.roles)) == toset(["web", "agent", "parser", "updater", "export"])
    error_message = "Define exactly web, agent, parser, updater and export roles."
  }
  validation {
    condition     = alltrue([for role in values(var.roles) : can(regex("^[a-z0-9][a-z0-9./:_-]+@sha256:[a-f0-9]{64}$", role.image)) && role.cpus > 0 && role.memory_mb >= 256 && role.state_disk_gb >= 10 && floor(role.state_disk_gb) == role.state_disk_gb])
    error_message = "Every role needs an immutable OCI digest and positive CPU/memory/disk limits."
  }
  validation {
    condition = alltrue(flatten([for role in values(var.roles) : [
      for name, file in role.config_files :
      can(regex("^[A-Za-z0-9][A-Za-z0-9_./-]*$", name)) && !strcontains(name, "..") &&
      can(regex("^[A-Za-z0-9][A-Za-z0-9_./-]*$", file.object)) && !strcontains(file.object, "..") &&
      can(regex("^[1-9][0-9]*$", file.generation)) && can(regex("^[a-f0-9]{64}$", file.sha256))
    ]]))
    error_message = "Config files need safe relative paths, immutable numeric GCS generations and sha256 checksums."
  }
  validation {
    condition = alltrue(flatten([for role in values(var.roles) : [
      for name, ref in role.secret_files : can(regex("^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$", name)) &&
      can(regex("^projects/[a-z][a-z0-9-]+/secrets/[A-Za-z0-9_-]+/versions/[1-9][0-9]*$", ref))
    ]]))
    error_message = "Secrets are filename -> pinned Secret Manager version references, never secret values."
  }
  validation {
    condition = alltrue(flatten([for name, role in var.roles : [for ref in values(role.secret_files) :
      name != "parser" && contains(values(name == "web" ? var.foundation.secret_refs.api : var.foundation.secret_refs.worker), ref)
    ]])) && length(try(var.roles.parser.secret_files, {})) == 0
    error_message = "Use only foundation-granted secret references for the role identity. Parser cannot receive secrets."
  }
  validation {
    condition = alltrue([for name, role in var.roles :
      (name == "web" || (role.zone != null && contains(var.zones, role.zone))) &&
      alltrue([for arg in role.command : can(regex("^[A-Za-z0-9_./=:@,+%-]+$", arg))])
    ])
    error_message = "Worker zones must be selected from zones; command is a safe argv array, not shell text."
  }
  validation {
    condition     = alltrue([for name in ["web", "agent", "updater", "export"] : contains(keys(try(var.roles[name].config_files, {})), "tokenizer.json")])
    error_message = "Each application role must provide the qualified tokenizer.json artifact."
  }
  validation {
    condition = alltrue([for name, role in var.roles : role.renderer_archive == null ? name != "export" :
      name == "export" && contains(keys(role.config_files), role.renderer_archive.config_file) &&
      startswith(role.renderer_archive.config_file, "export-release/") && can(regex("^sha256:[a-f0-9]{64}$", role.renderer_archive.image_id))
    ])
    error_message = "Export needs a checksummed export-release archive and its exact renderer Docker image ID."
  }
}

variable "artifact_registry_repositories" {
  description = "Existing repositories holding role images. Additive repository-scoped reader grants."
  type        = set(object({ project = string, location = string, repository = string }))
  default     = []
}

variable "domains" {
  type        = list(string)
  description = "Public DNS names for Google's managed HTTPS certificate; point their A records to the output address."
  validation {
    condition     = length(var.domains) > 0 && alltrue([for domain in var.domains : can(regex("^[a-z0-9][a-z0-9.-]+\\.[a-z]{2,}$", domain))])
    error_message = "Supply one or more public DNS hostnames."
  }
}

variable "web_replicas" {
  type        = number
  default     = 2
  description = "Regional Web capacity; at least one instance per configured zone."
  validation {
    condition     = var.web_replicas >= length(var.zones) && floor(var.web_replicas) == var.web_replicas
    error_message = "Use an integer web_replicas >= the number of zones."
  }
}

variable "source_cas_nfs" {
  description = "Existing private NFSv4.1 source CAS shared by Web, source updater and readers. Business blobs remain GCS."
  type        = object({ address = string, export_path = string })
  validation {
    condition     = can(cidrnetmask("${var.source_cas_nfs.address}/32")) && can(regex("^(10\\.|192\\.168\\.|172\\.(1[6-9]|2[0-9]|3[01])\\.)", var.source_cas_nfs.address)) && can(regex("^/[A-Za-z0-9_/-]+$", var.source_cas_nfs.export_path)) && !strcontains(var.source_cas_nfs.export_path, "..")
    error_message = "Supply a private NFS IPv4 address and safe absolute export path."
  }
}

variable "initialize_blank_state_disks" {
  type        = bool
  default     = false
  description = "Explicit permission to format newly created, signature-free worker disks as ext4. Existing non-ext4 disks are rejected."
}

variable "labels" {
  type    = map(string)
  default = {}
}
