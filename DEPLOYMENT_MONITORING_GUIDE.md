# 🚀 部署與監控完整指南

## 📋 目錄
- [快速開始](#快速開始)
- [本地部署](#本地部署)
- [容器化部署](#容器化部署)
- [雲端部署](#雲端部署)
- [監控設置](#監控設置)
- [故障排除](#故障排除)

---

## 🚀 快速開始

### 1. 一鍵執行完整流水線

```bash
# 乾跑模式（測試流程）
poetry run python scripts/automation/full_mlops_pipeline.py --dry-run

# 實際執行
poetry run python scripts/automation/full_mlops_pipeline.py --config scripts/automation/pipeline_config.json
```

### 2. 分步驟執行

```bash
# 步驟 1: 訓練模型
poetry run python application/training/pipelines/iris_training_pipeline.py --config application/training/configs/iris_config.json

# 步驟 2: 驗證模型
poetry run python application/validation/model_validation/validate_model.py --model-path application/registry/model_registry/ --threshold 0.95

# 步驟 3: 啟動服務
make run
```

---

## 🏠 本地部署

### 開發環境部署

```bash
# 1. 安裝依賴
make install

# 2. 訓練模型
poetry run python application/training/pipelines/iris_training_pipeline.py

# 3. 啟動 BentoML 服務 (推薦使用 Makefile)
make run

# 或手動啟動 (無警告版本)
cd application/inference/services
PYTHONWARNINGS="ignore" poetry run bentoml serve iris_service.py:svc --reload --port 3000
```

### 生產環境部署

```bash
# 1. 建立 BentoML 服務
cd application/inference/services
poetry run bentoml build

# 2. 啟動服務
BENTO_TAG=$(poetry run bentoml list iris_classifier --output json | jq -r '.[0].tag')
poetry run bentoml serve $BENTO_TAG --port 3000
```

### 服務測試

```bash
# 功能測試
poetry run python application/inference/services/test_service.py

# 負載測試
poetry run python tests/integration/test_load_performance.py --concurrent 10 --requests 20

# 健康檢查
curl http://localhost:3000/health_check
```

---

## 🐳 容器化部署

### Docker 基礎部署

#### 1. 建立 Docker 映像

```bash
# 使用 BentoML 建立
cd application/inference/services
poetry run bentoml build
BENTO_TAG=$(poetry run bentoml list iris_classifier --output json | jq -r '.[0].tag')
poetry run bentoml containerize $BENTO_TAG --platform linux/amd64
```

#### 2. 運行容器

```bash
# 獲取映像名稱
DOCKER_IMAGE=$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep iris_classifier | head -1)

# 運行容器
docker run -p 3000:3000 $DOCKER_IMAGE

# 背景運行
docker run -d -p 3000:3000 --name iris-service $DOCKER_IMAGE
```

#### 3. 容器管理

```bash
# 查看日誌
docker logs iris-service

# 進入容器
docker exec -it iris-service /bin/bash

# 停止容器
docker stop iris-service

# 移除容器
docker rm iris-service
```

### Docker Compose 部署

創建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  iris-service:
    build:
      context: .
      dockerfile: infrastructure/deployment/containers/docker/Dockerfile
    ports:
      - "3000:3000"
    environment:
      - BENTOML_PORT=3000
      - BENTOML_HOST=0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health_check"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./infrastructure/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

```bash
# 啟動服務
docker-compose up -d

# 查看服務狀態
docker-compose ps

# 停止服務
docker-compose down
```

---

## ☁️ 雲端部署

### AWS 部署

#### 1. AWS ECS 部署

```bash
# 安裝 AWS CLI
pip install awscli

# 配置 AWS 憑證
aws configure

# 推送映像到 ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com

# 標記並推送映像
docker tag iris_classifier:latest your-account.dkr.ecr.us-east-1.amazonaws.com/iris-classifier:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/iris-classifier:latest
```

創建 ECS 任務定義 `infrastructure/deployment/cloud/aws/task-definition.json`：

```json
{
  "family": "iris-classifier",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::your-account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "iris-classifier",
      "image": "your-account.dkr.ecr.us-east-1.amazonaws.com/iris-classifier:latest",
      "portMappings": [
        {
          "containerPort": 3000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/iris-classifier",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:3000/health_check || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### 2. AWS Lambda 部署

```bash
# 安裝 Serverless Framework
npm install -g serverless

# 創建 serverless.yml
cat > serverless.yml << EOF
service: iris-classifier-lambda

provider:
  name: aws
  runtime: python3.9
  stage: prod
  region: us-east-1

functions:
  predict:
    handler: lambda_handler.predict
    events:
      - http:
          path: predict
          method: post
          cors: true
    timeout: 30
    memorySize: 512

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
EOF

# 部署
sls deploy
```

### Google Cloud Platform 部署

#### 1. Cloud Run 部署

```bash
# 安裝 gcloud CLI
# https://cloud.google.com/sdk/docs/install

# 認證
gcloud auth login
gcloud config set project your-project-id

# 建立並推送映像
gcloud builds submit --tag gcr.io/your-project-id/iris-classifier

# 部署到 Cloud Run
gcloud run deploy iris-classifier \
  --image gcr.io/your-project-id/iris-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 3000 \
  --memory 1Gi \
  --cpu 1
```

#### 2. GKE 部署

創建 `infrastructure/deployment/cloud/gcp/deployment.yaml`：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iris-classifier
  template:
    metadata:
      labels:
        app: iris-classifier
    spec:
      containers:
      - name: iris-classifier
        image: gcr.io/your-project-id/iris-classifier:latest
        ports:
        - containerPort: 3000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health_check
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health_check
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: iris-classifier-service
spec:
  selector:
    app: iris-classifier
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

```bash
# 部署到 GKE
kubectl apply -f infrastructure/deployment/cloud/gcp/deployment.yaml

# 查看部署狀態
kubectl get deployments
kubectl get services
kubectl get pods
```

### Azure 部署

#### 1. Container Instances 部署

```bash
# 安裝 Azure CLI
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# 登入 Azure
az login

# 創建資源群組
az group create --name iris-classifier-rg --location eastus

# 部署容器
az container create \
  --resource-group iris-classifier-rg \
  --name iris-classifier \
  --image your-registry/iris-classifier:latest \
  --ports 3000 \
  --dns-name-label iris-classifier-unique \
  --memory 1 \
  --cpu 1
```

---

## 📊 監控設置

### 1. Prometheus 監控

創建 `infrastructure/monitoring/prometheus/prometheus.yml`：

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'iris-classifier'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'bentoml-service'
    static_configs:
      - targets: ['localhost:3000']
    metrics_path: '/metrics'
    scrape_interval: 10s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

創建告警規則 `infrastructure/monitoring/prometheus/alert_rules.yml`：

```yaml
groups:
  - name: iris_classifier_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(ml_prediction_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighLatency
        expr: histogram_quantile(0.95, ml_prediction_duration_seconds) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ $value }}s"

      - alert: LowAccuracy
        expr: ml_model_accuracy < 0.9
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy is low"
          description: "Model accuracy is {{ $value }}"
```

### 2. Grafana 儀表板

啟動 Grafana 並導入儀表板：

```bash
# 啟動 Grafana
docker run -d -p 3001:3000 --name grafana grafana/grafana:latest

# 訪問 http://localhost:3001
# 預設帳號: admin/admin

# 添加 Prometheus 資料源
# URL: http://localhost:9090
```

創建儀表板 JSON 配置：

```json
{
  "dashboard": {
    "title": "Iris Classifier Monitoring",
    "panels": [
      {
        "title": "Prediction Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_predictions_total[5m])",
            "legendFormat": "{{class}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, ml_prediction_duration_seconds)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, ml_prediction_duration_seconds)",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_prediction_errors_total[5m])",
            "legendFormat": "{{error_type}}"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "singlestat",
        "targets": [
          {
            "expr": "ml_model_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      }
    ]
  }
}
```

### 3. 日誌監控

使用 ELK Stack 進行日誌監控：

```bash
# 啟動 Elasticsearch, Logstash, Kibana
docker-compose -f infrastructure/monitoring/elk/docker-compose.yml up -d
```

ELK `docker-compose.yml`：

```yaml
version: '3.7'
services:
  elasticsearch:
    image: elasticsearch:7.14.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"

  logstash:
    image: logstash:7.14.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5000:5000"
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:7.14.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_URL=http://elasticsearch:9200
    depends_on:
      - elasticsearch
```

---

## 🔧 故障排除

### 常見問題

#### 1. 模型載入失敗

```bash
# 檢查模型文件
ls -la application/registry/model_registry/

# 檢查 MLflow 模型
poetry run python -c "import mlflow; print(mlflow.search_experiments())"

# 重新訓練模型
poetry run python application/training/pipelines/iris_training_pipeline.py
```

#### 2. 服務無法啟動

```bash
# 檢查埠號占用
netstat -tulpn | grep :3000

# 檢查 BentoML 服務列表
poetry run bentoml list

# 查看詳細錯誤
poetry run bentoml serve iris_service.py:svc --reload --debug
```

#### 3. 性能問題

```bash
# 運行性能測試
poetry run python tests/integration/test_load_performance.py --concurrent 1 --requests 10

# 檢查資源使用
docker stats iris-service

# 調整資源配置
# 編輯 bentofile.yaml 或 deployment.yaml
```

#### 4. 監控指標缺失

```bash
# 檢查 Prometheus 目標
curl http://localhost:9090/targets

# 檢查指標端點
curl http://localhost:8001/metrics

# 重啟監控組件
docker-compose restart prometheus grafana
```

#### 5. BentoML 特定問題

```bash
# 檢查 BentoML 版本兼容性
poetry run bentoml --version

# 常見 API 錯誤修復
# 錯誤: AttributeError: module 'bentoml' has no attribute 'service'
# 解決: 改用 @bentoml.Service (大寫 S)

# 錯誤: TypeError: Service.__init__() got an unexpected keyword argument 'resources'
# 解決: 將 resources 參數移到 @bentoml.api 裝飾器

# 錯誤: Attribute "IrisClassifier" not found in module
# 解決: 檢查服務實例名稱，使用 iris_service.py:svc

# 抑制警告訊息
PYTHONWARNINGS="ignore" poetry run bentoml serve iris_service.py:svc --reload
```

### 6. Makefile 命令問題

```bash
# 查看所有可用命令
make help

# 檢查 Poetry 環境
poetry env info

# 重新安裝依賴
make clean && make install

# 檢查 GPU 配置
make checkgpu

# 構建服務包
make bento-build

# 創建容器
make containerize

# 啟動服務
make run
```

**常見 Makefile 錯誤：**

#### `make install` 失敗
```bash
# 檢查 Poetry 鎖定文件
ls -la poetry.lock

# 重新鎖定依賴
poetry lock --no-update

# 清理緩存後重新安裝
poetry cache clear --all pypi
make install
```

#### `make run` 服務啟動失敗
```bash
# 檢查模型是否已載入
poetry run bentoml models list

# 檢查服務構建狀態
make bento-build

# 查看詳細日誌
tail -f bentoml_service.log
```

#### `make containerize` 失敗
```bash
# 檢查 Docker 是否運行
docker ps

# 檢查服務是否已構建
make bento-build

# 查看容器日誌
docker logs <container_id>
```

### 除錯工具

#### 1. 日誌查看

```bash
# BentoML 服務日誌
poetry run bentoml logs iris_classifier:latest

# Docker 容器日誌
docker logs iris-service

# Kubernetes Pod 日誌
kubectl logs deployment/iris-classifier
```

#### 2. 健康檢查

```bash
# 服務健康檢查
curl http://localhost:3000/health_check

# 詳細指標
curl http://localhost:3000/get_metrics_summary?hours=1

# Prometheus 指標
curl http://localhost:8001/metrics
```

#### 3. 效能分析

```bash
# 記憶體使用
poetry run python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# CPU 使用
top -p $(pgrep -f bentoml)
```

---

## 🎯 最佳實踐

### 1. 安全性

- 使用 HTTPS
- 實施 API 金鑰認證
- 定期更新依賴項
- 掃描容器漏洞

### 2. 可擴展性

- 使用負載均衡器
- 實施水平擴展
- 設置自動擴展
- 監控資源使用

### 3. 可靠性

- 實施健康檢查
- 設置重試機制
- 配置監控告警
- 定期備份模型

### 4. 效能優化

- 使用模型量化
- 實施批次處理
- 調整並發設置
- 使用快取機制

---

## 📞 支援

遇到問題？

1. 查看 [故障排除](#故障排除) 部分
2. 檢查專案 [README.md](README.md)
3. 查閱 [MLOps 完整教學](MLOPS_COMPLETE_TUTORIAL.md)
4. 提交 Issue 到專案儲存庫

**恭喜！您現在擁有完整的 MLOps 部署和監控系統！** 🎉