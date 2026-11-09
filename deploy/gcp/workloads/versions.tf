terraform {
  required_version = "= 1.17.1"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "= 8.10.0"
    }
  }
}

provider "google" {
  project = var.foundation.project_id
  region  = var.foundation.region
}
