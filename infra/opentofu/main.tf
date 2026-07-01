# OPTIONAL single-VM deploy (OpenTofu, MPL-2.0 — not Terraform/BSL).
# Provisions one cloud VM and runs `docker compose --profile core up -d`.
# This is a stub; fill in your provider + image. Cloud-agnostic by design.

terraform {
  required_version = ">= 1.8"
  # required_providers { <your-cloud> = { ... } }  # e.g. aws / google / azurerm / hcloud
}

variable "instance_type" {
  type    = string
  default = "t3.large" # ~8GB RAM comfortably runs the core profile
}

variable "ssh_key_name" { type = string }

# TODO(phase-6): provider block + a compute instance whose user-data:
#   1. installs docker + compose plugin
#   2. clones this repo (or pulls a built image)
#   3. writes .env from a secrets manager (NEVER bake secrets into the image)
#   4. runs: docker compose --profile core up -d
#
# Scale-out path: swap this for the Helm chart under infra/helm/ (K8s).

output "next_steps" {
  value = "Stub. See docs/ADAPTING.md for the deploy story before applying."
}
