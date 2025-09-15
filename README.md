# 🚀 MLOps Template - 完整的機器學習運維系統

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue.svg)](https://python-poetry.org/)
[![BentoML](https://img.shields.io/badge/serving-bentoml-green.svg)](https://bentoml.org/)
[![Docker](https://img.shields.io/badge/container-docker-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **企業級 MLOps 範本**：從實驗到生產的完整機器學習生命週期管理系統

## 🎯 專案特色

- 🏗️ **系統架構分層** - Domain/Application/Infrastructure 清晰分離
- 📦 **Poetry 依賴管理** - 精確版本控制，告別依賴地獄
- 🤖 **多框架支援** - scikit-learn、PyTorch、TensorFlow、Hugging Face
- 🚀 **一鍵自動化** - 從訓練到部署的完整流水線
- 📊 **企業級監控** - Prometheus + Grafana + 自定義指標
- ☁️ **多雲支援** - AWS、GCP、Azure 部署配置
- 🧪 **完整測試** - 單元測試、整合測試、負載測試
- 📖 **詳細文檔** - 完整教學和 API 文檔

## 🚀 快速開始

### 一鍵啟動

```bash
# 下載專案
git clone <your-repo-url>
cd mlops-template

# 一鍵執行完整 MLOps 流水線
bash scripts/quickstart.sh
```

這將自動：
1. ✅ 檢查系統需求和安裝 Poetry
2. 📦 安裝所有依賴（包含 GPU 支援）
3. 🔧 驗證 GPU 環境
4. 🏃 訓練 Iris 分類模型
5. ✅ 驗證模型品質
6. 🚀 建立和啟動 BentoML 服務
7. 🧪 執行功能和負載測試
8. 📊 啟動監控指標

### 分步驟執行

```bash
# 安裝依賴
make install

# 檢查 GPU 支援
make checkgpu

# 訓練模型
poetry run python application/training/pipelines/iris_training_pipeline.py

# 啟動服務
make run
```

## 📁 專案架構

```
mlops-template/
├── 🏢 domain/              # 領域層 - 核心業務邏輯
│   ├── data/              # 資料管理和處理
│   ├── models/            # 各種 ML 模型
│   │   ├── traditional/   # scikit-learn 模型
│   │   ├── deep_learning/ # PyTorch/TensorFlow 模型
│   │   ├── generative/    # 生成式 AI (LLM/Whisper)
│   │   └── specialized/   # 專業領域模型 (NLP/CV)
│   └── experiments/       # Jupyter 實驗和研究
├── 🔧 application/         # 應用層 - 業務應用
│   ├── training/          # 自動化訓練流水線
│   ├── inference/         # BentoML 推論服務
│   ├── validation/        # 模型驗證系統
│   └── registry/          # 模型註冊表和製品管理
├── 🏗️ infrastructure/      # 基礎設施層
│   ├── deployment/        # Docker/Kubernetes 部署
│   ├── monitoring/        # Prometheus/Grafana 監控
│   └── cicd/             # GitHub Actions CI/CD
├── 🔄 shared/             # 共享層
│   ├── utils/            # 通用工具程式庫
│   └── configs/          # 配置管理
├── 🧪 tests/              # 測試套件
├── 📖 examples/           # 範例和教程
└── 📜 scripts/            # 自動化腳本
```

## 🎓 學習路徑

### 🔰 初學者路徑
1. **環境設置** → [快速開始](#快速開始)
2. **第一個模型** → 運行 Iris 分類範例
3. **API 使用** → 測試 BentoML 服務
4. **基礎監控** → 查看 Prometheus 指標

### 🎯 進階路徑
1. **系統架構** → 閱讀 [系統設計文檔](SYSTEM_DESIGN_RESTRUCTURE.md)
2. **完整教學** → 跟隨 [MLOps 完整教學](MLOPS_COMPLETE_TUTORIAL.md)
3. **自定義模型** → 開發您的模型
4. **部署指南** → [部署與監控指南](DEPLOYMENT_MONITORING_GUIDE.md)

### 🚀 專家路徑
1. **全流程自動化** → 使用 `scripts/automation/`
2. **雲端部署** → AWS/GCP/Azure 部署
3. **CI/CD 整合** → GitHub Actions 工作流
4. **監控優化** → 進階監控和告警

## 🛠️ 核心功能

### 機器學習模型支援

| 框架 | 位置 | 說明 |
|------|------|------|
| **scikit-learn** | `domain/models/traditional/` | 傳統機器學習 |
| **PyTorch** | `domain/models/deep_learning/pytorch/` | 深度學習 |
| **TensorFlow** | `domain/models/deep_learning/tensorflow/` | 深度學習 |
| **Hugging Face** | `domain/models/generative/llm/` | 生成式 AI |
| **專業模型** | `domain/models/specialized/` | NLP、CV、時間序列 |

### 自動化工具

```bash
# 完整流水線自動化
poetry run python scripts/automation/full_mlops_pipeline.py

# 模型驗證
poetry run python application/validation/model_validation/validate_model.py

# 負載測試
poetry run python tests/integration/test_load_performance.py
```

### 監控系統

- **模型指標** - 準確率、延遲、吞吐量
- **系統監控** - CPU、記憶體、GPU 使用率
- **業務指標** - 預測分布、資料漂移檢測
- **告警系統** - Prometheus 規則和通知

## 🌐 API 使用

### 健康檢查
```bash
curl http://localhost:3000/health_check
```

### 單個預測
```bash
curl -X POST http://localhost:3000/classify_json \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### 批次預測
```bash
curl -X POST http://localhost:3000/classify_json \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
      {"sepal_length": 6.2, "sepal_width": 2.9, "petal_length": 4.3, "petal_width": 1.3}
    ]
  }'
```

## 🐳 部署選項

### 本地部署
```bash
make run
```

### Docker 部署
```bash
# 建立容器
make containerize

# 運行容器
docker run -p 3000:3000 iris_classifier:latest
```

### 雲端部署
```bash
# AWS ECS
aws ecs create-service --service-name iris-classifier

# Google Cloud Run
gcloud run deploy iris-classifier --image gcr.io/project/iris-classifier

# Azure Container Instances
az container create --name iris-classifier
```

## 📊 監控儀表板

啟動完整監控棧：

```bash
docker-compose -f infrastructure/monitoring/docker-compose.yml up -d
```

訪問監控介面：
- **服務 API**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin)

## 🧪 測試

### 單元測試
```bash
make test
```

### 整合測試
```bash
poetry run python tests/integration/test_load_performance.py
```

### 程式碼品質
```bash
make refactor  # format + lint
```

## 📚 完整文檔

| 文檔 | 說明 |
|------|------|
| [🎓 MLOps 完整教學](MLOPS_COMPLETE_TUTORIAL.md) | 6 章節完整教學 |
| [🚀 部署與監控指南](DEPLOYMENT_MONITORING_GUIDE.md) | 本地到雲端部署 |
| [🎯 快速入門指南](GETTING_STARTED.md) | 分層學習路徑 |
| [🏗️ 系統設計文檔](SYSTEM_DESIGN_RESTRUCTURE.md) | 架構設計原理 |
| [📦 Poetry 使用指南](POETRY_QUICKSTART.md) | 依賴管理詳解 |
| [🔄 遷移報告](MIGRATION_TO_POETRY.md) | 從 pip 到 Poetry |

## ⚙️ 系統需求

### 最低需求
- **Python**: 3.9+
- **記憶體**: 4GB
- **硬碟**: 5GB 可用空間

### 建議需求
- **Python**: 3.9+
- **記憶體**: 8GB+
- **硬碟**: 10GB+ 可用空間
- **GPU**: NVIDIA GPU (CUDA 11.8+) 用於深度學習

### 相依工具
- **Poetry**: 依賴管理
- **Docker**: 容器化 (可選)
- **Git**: 版本控制

## 🔧 故障排除

### 常見問題

#### Poetry 安裝失敗
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

#### GPU 不可用
```bash
# 檢查 NVIDIA 驅動
nvidia-smi

# 重新安裝 GPU 版本
poetry run pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 埠號衝突
```bash
# 使用不同埠號
poetry run bentoml serve iris_service:IrisClassifier --port 3001
```

#### 模型載入失敗
```bash
# 重新訓練模型
poetry run python application/training/pipelines/iris_training_pipeline.py
```

更多故障排除請查看 [部署與監控指南](DEPLOYMENT_MONITORING_GUIDE.md#故障排除)

## 🤝 貢獻

我們歡迎所有形式的貢獻！

1. 🍴 Fork 專案
2. 🌿 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 💾 提交變更 (`git commit -m 'Add AmazingFeature'`)
4. 📤 推送到分支 (`git push origin feature/AmazingFeature`)
5. 🔀 開啟 Pull Request

## 📄 授權

本專案採用 MIT 授權 - 查看 [LICENSE](LICENSE) 文件了解詳情

## 🙏 致謝

- [BentoML](https://bentoml.org/) - 模型服務框架
- [Poetry](https://python-poetry.org/) - Python 依賴管理
- [MLflow](https://mlflow.org/) - ML 生命週期管理
- [Prometheus](https://prometheus.io/) - 監控系統
- [Grafana](https://grafana.com/) - 可視化儀表板

## 📞 支援

- 📖 查看 [文檔](#完整文檔)
- 🐛 [提交 Issue](../../issues)
- 💬 [參與討論](../../discussions)
- 📧 聯繫維護者

---

**⭐ 如果這個專案對您有幫助，請給我們一個 star！**

**🚀 立即開始您的 MLOps 之旅！**