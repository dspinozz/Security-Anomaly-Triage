# Terraform Infrastructure for Security-Anomaly-Triage

This Terraform configuration deploys the Security-Anomaly-Triage application to AWS using ECS Fargate.

## Architecture

- **VPC**: Custom VPC with public and private subnets across multiple AZs
- **ECR**: Container registry for Docker images
- **ECS Fargate**: Container orchestration (serverless)
- **Application Load Balancer**: HTTP/HTTPS load balancing
- **S3**: Storage for ML models
- **CloudWatch**: Logging and monitoring

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform >= 1.0 installed
3. Docker installed (for building images)

## Quick Start

### 1. Initialize Terraform

```bash
cd terraform
terraform init
```

### 2. Review and Customize Variables

Edit `terraform.tfvars` (create if needed):

```hcl
aws_region      = "us-east-1"
environment     = "dev"
image_tag       = "latest"
ecs_cpu         = 2048
ecs_memory      = 4096
ecs_desired_count = 2
```

### 3. Validate Configuration

```bash
terraform validate
terraform plan
```

### 4. Deploy Infrastructure

```bash
terraform apply
```

### 5. Build and Push Docker Image

After infrastructure is created:

```bash
# Get ECR login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ECR_REPO_URL>

# Build image
docker build -t security-anomaly-triage:latest ..

# Tag image
docker tag security-anomaly-triage:latest <ECR_REPO_URL>:latest

# Push image
docker push <ECR_REPO_URL>:latest
```

### 6. Access the API

The ALB DNS name will be in the outputs:

```bash
terraform output alb_dns_name
```

Then access: `http://<alb_dns_name>/health`

## Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `aws_region` | AWS region | `us-east-1` |
| `environment` | Environment (dev/staging/prod) | `dev` |
| `vpc_cidr` | VPC CIDR block | `10.0.0.0/16` |
| `image_tag` | Docker image tag | `latest` |
| `ecs_cpu` | CPU units (1024 = 1 vCPU) | `2048` |
| `ecs_memory` | Memory in MB | `4096` |
| `ecs_desired_count` | Number of tasks | `2` |

## Outputs

- `ecr_repository_url`: ECR repository URL for pushing images
- `alb_dns_name`: Application Load Balancer DNS name
- `api_endpoint`: Full API endpoint URL
- `models_bucket_name`: S3 bucket for ML models
- `cloudwatch_log_group`: CloudWatch log group name

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

## Notes

- Models should be uploaded to the S3 bucket before deployment
- The application expects models in `models/trained/` directory
- Health check endpoint: `/health`
- Application runs on port 8001
