terraform {
  required_version = "= 1.17.1"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "= 8.10.0"
    }
  }
}

# No credentials, data sources, API-enabling resources, or remote backend.
# Real state/IAM/bootstrap choices require a separate deployment review.
provider "google" {
  project = var.project_id
  region  = var.region
}
