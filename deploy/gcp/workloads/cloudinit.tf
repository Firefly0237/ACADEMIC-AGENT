locals {
  bootstrap_scripts = { for name, role in var.roles : name => templatefile("${path.module}/templates/bootstrap.sh.tftpl", {
    role_name       = name
    role            = role
    config_bucket   = var.config_bucket
    bootstrap_image = var.bootstrap_image
    proxy_image     = var.sql_proxy_image
    initialize_disk = var.initialize_blank_state_disks
    source_nfs      = var.source_cas_nfs
    source_volume   = local.source_volume
    registries = distinct([for image in [role.image, var.bootstrap_image, var.sql_proxy_image] :
      split("/", image)[0] if can(regex("(^|\\.)gcr\\.io$|^[a-z0-9-]+-docker\\.pkg\\.dev$", split("/", image)[0]))
    ])
  }) }
  fetch_scripts = { for name, role in var.roles : name => templatefile("${path.module}/templates/fetch.sh.tftpl", {
    role          = role
    config_bucket = var.config_bucket
  }) }
  start_scripts = { for name, role in var.roles : name => templatefile("${path.module}/templates/start-role.sh.tftpl", {
    role_name     = name
    role          = role
    source_volume = local.source_volume
  }) }
  cloud_config = { for name, role in var.roles : name => "#cloud-config\n${yamlencode({
    write_files = concat([
      {
        path        = "/var/lib/citequest/bootstrap.sh"
        permissions = "0700"
        owner       = "root:root"
        encoding    = "b64"
        content     = base64encode(local.bootstrap_scripts[name])
      },
      {
        path        = "/var/lib/citequest/fetch.sh"
        permissions = "0700"
        owner       = "root:root"
        encoding    = "b64"
        content     = base64encode(local.fetch_scripts[name])
      },
      {
        path        = "/var/lib/citequest/start-role.sh"
        permissions = "0700"
        owner       = "root:root"
        encoding    = "b64"
        content     = base64encode(local.start_scripts[name])
      },
      {
        path        = "/etc/systemd/system/citequest-bootstrap.service"
        permissions = "0644"
        owner       = "root:root"
        content     = <<-UNIT
          [Unit]
          Description=Prepare immutable CiteQuest ${name} release
          Wants=network-online.target
          After=network-online.target docker.service
          Requires=docker.service
          Before=citequest-role.service citequest-sql-proxy.service
          [Service]
          Type=oneshot
          RemainAfterExit=yes
          Environment=DOCKER_CONFIG=/var/lib/citequest/docker
          ExecStart=/bin/bash /var/lib/citequest/bootstrap.sh
          TimeoutStartSec=1800
          [Install]
          WantedBy=multi-user.target
        UNIT
      }
    ], name == "parser" ? [] : [
      {
        path        = "/etc/systemd/system/citequest-sql-proxy.service"
        permissions = "0644"
        owner       = "root:root"
        content     = <<-UNIT
          [Unit]
          Description=Cloud SQL Auth Proxy private-IP loopback transport
          After=citequest-bootstrap.service docker.service
          Requires=citequest-bootstrap.service docker.service
          Before=citequest-role.service
          [Service]
          Environment=DOCKER_CONFIG=/var/lib/citequest/docker
          ExecStart=/usr/bin/docker run --rm --name=cq-sql-proxy --network=host --read-only --cap-drop=ALL --security-opt=no-new-privileges --log-driver=local --log-opt=max-size=10m --log-opt=max-file=3 ${var.sql_proxy_image} --private-ip --address=127.0.0.1 --port=5432 --structured-logs ${var.foundation.postgres.connection_name}
          ExecStop=/usr/bin/docker stop --time=60 cq-sql-proxy
          Restart=on-failure
          RestartSec=5
          TimeoutStopSec=75
          [Install]
          WantedBy=multi-user.target
        UNIT
      },
      {
        path        = "/etc/systemd/system/citequest-role.service"
        permissions = "0644"
        owner       = "root:root"
        content     = <<-UNIT
          [Unit]
          Description=CiteQuest ${name} runtime
          After=citequest-bootstrap.service citequest-sql-proxy.service
          Requires=citequest-bootstrap.service
          Wants=citequest-sql-proxy.service
          [Service]
          Environment=DOCKER_CONFIG=/var/lib/citequest/docker
          ExecStart=/bin/bash /var/lib/citequest/start-role.sh
          ExecStop=/usr/bin/docker stop --time=${name == "web" ? "300" : "-1"} cq-${name}
          Restart=${name == "web" ? "on-failure" : "no"}
          RestartSec=5
          KillMode=none
          TimeoutStopSec=${name == "web" ? "315" : "infinity"}
          [Install]
          WantedBy=multi-user.target
        UNIT
      }
    ])
    # COS /etc is recreated on each boot; cloud-init reconstructs these units.
    runcmd = concat([
      ["systemctl", "daemon-reload"],
      ["systemctl", "start", "citequest-bootstrap.service"]
    ], name == "parser" ? [] : [
      ["systemctl", "start", "citequest-sql-proxy.service"],
      ["systemctl", "start", "citequest-role.service"]
    ])
  })}" }
}
