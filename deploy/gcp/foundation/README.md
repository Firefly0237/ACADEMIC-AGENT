# Google Cloud 基础设施

Terraform模块为CiteQuest准备私有网络、数据库、对象存储和服务身份。

## 资源

- 自定义VPC、子网与Private Services Access
- Cloud SQL PostgreSQL 17
- 私有GCS bucket
- Web/API、worker与检索服务账号
- Cloud SQL IAM登录及Secret Manager访问配置

模块输出`foundation_contract`，包含网络、数据库、存储和服务身份信息，供应用部署使用。

## 使用

安装Terraform 1.17.1，并配置Google Cloud项目和Terraform凭据。模块使用锁定的Google provider。

```bash
cd deploy/gcp/foundation
cp terraform.tfvars.example.json terraform.tfvars.json
```

在`terraform.tfvars.json`中填写：

| 参数 | 用途 |
| --- | --- |
| `project_id`、`region` | 项目与区域 |
| `subnet_cidr`、`private_services_cidr` | 应用子网与数据库私网地址段 |
| `blob_bucket_name` | 对象存储名称 |
| `sql_instance_name`、`sql_tier` | 数据库实例与规格 |
| `sql_disk_size_gb`、`sql_disk_limit_gb` | 数据库磁盘配置 |
| `api_secret_refs`、`worker_secret_refs` | 角色所用Secret版本引用 |

初始化并生成部署计划：

```bash
terraform init -lockfile=readonly
terraform fmt -check
terraform validate
terraform plan -var-file=terraform.tfvars.json -out=foundation.tfplan
```

应用部署使用[核心Compose框架](../../core/compose.yaml)，配置步骤见[快速开始](../../../docs/quickstart.md)。
