variable "project_id" {
  type        = string
  description = "Existing GCP project; this module neither creates it nor enables APIs."
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "Use an existing 6-30 character project ID, not a name or project number."
  }
}

variable "region" {
  type        = string
  description = "Single region for subnet, Cloud SQL and business bucket; availability remains a cloud gate."
  validation {
    condition     = can(regex("^[a-z]+-[a-z]+[0-9]+$", var.region))
    error_message = "Use a GCP region such as asia-east1, not a zone."
  }
}

variable "name_prefix" {
  type        = string
  description = "Unique environment prefix used for resources and dedicated runtime identities."
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,18}[a-z0-9]$", var.name_prefix))
    error_message = "Use a lowercase 4-20 character prefix ending in a letter or digit."
  }
}

variable "subnet_cidr" {
  type        = string
  description = "Non-overlapping IPv4 subnet; /26 or larger leaves a usable Direct VPC handoff."
  validation {
    condition     = can(cidrnetmask(var.subnet_cidr)) && try(tonumber(split("/", var.subnet_cidr)[1]) >= 16 && tonumber(split("/", var.subnet_cidr)[1]) <= 26, false)
    error_message = "Supply an IPv4 /16 through /26 subnet; verify all network overlaps before cloud deployment."
  }
  validation {
    condition     = can(regex("^(10\\.|192\\.168\\.|172\\.(1[6-9]|2[0-9]|3[01])\\.)", var.subnet_cidr)) && try(var.subnet_cidr == "${cidrhost(var.subnet_cidr, 0)}/${tonumber(split("/", var.subnet_cidr)[1])}", false)
    error_message = "Use a canonical RFC1918 network address, with host bits zero."
  }
}

variable "private_services_cidr" {
  type        = string
  description = "Distinct RFC1918 allocation for Private Services Access, not a VM/Cloud Run subnet."
  validation {
    condition     = can(cidrnetmask(var.private_services_cidr)) && try(tonumber(split("/", var.private_services_cidr)[1]) >= 16 && tonumber(split("/", var.private_services_cidr)[1]) <= 20, false)
    error_message = "Use a separate IPv4 /16 through /20 allocation; sizing/overlap require future cloud review."
  }
  validation {
    condition     = can(regex("^(10\\.|192\\.168\\.|172\\.(1[6-9]|2[0-9]|3[01])\\.)", var.private_services_cidr)) && try(var.private_services_cidr == "${cidrhost(var.private_services_cidr, 0)}/${tonumber(split("/", var.private_services_cidr)[1])}", false)
    error_message = "Use a canonical RFC1918 Private Services Access range, with host bits zero."
  }
  validation {
    # Neither canonical base may be inside the other network. Other existing
    # networks, routes, Shared VPC and organization constraints remain untested.
    condition = try(
      cidrhost("${cidrhost(var.private_services_cidr, 0)}/${split("/", var.subnet_cidr)[1]}", 0) != cidrhost(var.subnet_cidr, 0) &&
      cidrhost("${cidrhost(var.subnet_cidr, 0)}/${split("/", var.private_services_cidr)[1]}", 0) != cidrhost(var.private_services_cidr, 0),
      false
    )
    error_message = "The service subnet and Private Services Access allocation must not overlap."
  }
}

variable "blob_bucket_name" {
  type        = string
  description = "Globally unique business blob bucket name; no website or public ACL is configured."
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$", var.blob_bucket_name))
    error_message = "Use a globally unique 3-63 character lowercase bucket name with letters, digits and hyphens."
  }
}

variable "sql_instance_name" {
  type        = string
  description = "Cloud SQL instance ID; regional uniqueness and recent-name reuse must be checked remotely."
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,48}[a-z0-9]$", var.sql_instance_name))
    error_message = "Use a lowercase 3-50 character instance ID ending in a letter or digit."
  }
}

variable "sql_tier" {
  type        = string
  description = "Explicit Enterprise dedicated-core tier, chosen after capacity/cost review; no sizing claim."
  validation {
    condition     = can(regex("^db-custom-[1-9][0-9]*-[1-9][0-9]*$", var.sql_tier))
    error_message = "Use an explicit dedicated-core Enterprise db-custom-CPU-MEMORY tier."
  }
}

variable "sql_availability_type" {
  type        = string
  default     = "REGIONAL"
  description = "REGIONAL is the production recommendation. ZONAL is an explicit non-HA cost tradeoff."
  validation {
    condition     = contains(["REGIONAL", "ZONAL"], var.sql_availability_type)
    error_message = "Cloud SQL availability must be REGIONAL or ZONAL."
  }
}

variable "sql_disk_size_gb" {
  type        = number
  description = "Initial SSD size in GiB, not a production capacity estimate."
  validation {
    condition     = var.sql_disk_size_gb >= 20 && floor(var.sql_disk_size_gb) == var.sql_disk_size_gb
    error_message = "Use an integer initial disk size of at least 20 GiB."
  }
}

variable "sql_disk_limit_gb" {
  type        = number
  description = "Finite automatic-growth ceiling; exhausted space requires an explicit operational decision."
  validation {
    condition     = var.sql_disk_limit_gb >= var.sql_disk_size_gb && floor(var.sql_disk_limit_gb) == var.sql_disk_limit_gb
    error_message = "The integer growth limit must be at least the initial disk size; zero/unbounded is not allowed."
  }
}

variable "database_name" {
  type        = string
  default     = "citequest"
  description = "Empty application database only. SQL roles/schema/grants/migrations remain a separate bootstrap."
  validation {
    condition     = can(regex("^[a-z][a-z0-9_]{0,62}$", var.database_name))
    error_message = "Use a lowercase PostgreSQL database name."
  }
}

variable "api_secret_refs" {
  type = map(object({
    secret_id = string
    version   = string
  }))
  default     = {}
  description = "Existing same-project Secret Manager references by consumer name; numeric versions, never secret values."
  validation {
    condition     = alltrue([for ref in values(var.api_secret_refs) : can(regex("^[A-Za-z0-9_-]{1,255}$", ref.secret_id)) && can(regex("^[1-9][0-9]*$", ref.version))])
    error_message = "Provide secret IDs and pinned numeric version references only; latest or secret payloads are not inputs."
  }
}

variable "worker_secret_refs" {
  type = map(object({
    secret_id = string
    version   = string
  }))
  default     = {}
  description = "Existing same-project worker secret references; identities get accessor only on these secrets."
  validation {
    condition     = alltrue([for ref in values(var.worker_secret_refs) : can(regex("^[A-Za-z0-9_-]{1,255}$", ref.secret_id)) && can(regex("^[1-9][0-9]*$", ref.version))])
    error_message = "Provide secret IDs and pinned numeric versions; no latest or secret payloads."
  }
}

variable "labels" {
  type        = map(string)
  default     = {}
  description = "Non-secret operational labels. GCP label syntax/quota/organization policies remain deployment gates."
}
