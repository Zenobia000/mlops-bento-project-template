# 🚀 MLOps 專案入門指南

## 🎯 快速開始

### 一鍵執行完整流程

```bash
# Linux/Mac 用戶
bash scripts/quickstart.sh

# Windows 用戶 (Git Bash 或 WSL)
bash scripts/quickstart.sh

# 或使用 Python 腳本
poetry run python scripts/automation/full_mlops_pipeline.py
```

這個指令將自動：
1. ✅ 檢查系統需求
2. 📦 安裝所有依賴
3. 🔧 驗證 GPU 環境
4. 🏃 訓練 Iris 分類模型
5. ✅ 驗證模型品質
6. 🚀 建立 BentoML 服務
7. 🧪 執行功能和負載測試
8. 📊 啟動監控指標

### 分步驟執行

如果您想了解每個步驟：

```bash
# 步驟 1: 安裝依賴
make install

# 步驟 2: 檢查 GPU
make checkgpu

# 步驟 3: 訓練模型
poetry run python application/training/pipelines/iris_training_pipeline.py

# 步驟 4: 啟動服務
make run
```

## 📚 學習路徑

### 🔰 初學者路徑
1. **開始** → 閱讀 [README.md](README.md) 了解專案概述
2. **環境設置** → 使用 `bash scripts/quickstart.sh --install-only`
3. **第一個模型** → 運行 Iris 分類範例
4. **API 測試** → 使用 BentoML 服務 API

### 🎓 進階路徑
1. **系統架構** → 閱讀 [SYSTEM_DESIGN_RESTRUCTURE.md](SYSTEM_DESIGN_RESTRUCTURE.md)
2. **完整教學** → 跟隨 [MLOPS_COMPLETE_TUTORIAL.md](MLOPS_COMPLETE_TUTORIAL.md)
3. **自定義模型** → 在 `domain/models/` 中開發您的模型
4. **部署生產** → 使用 [DEPLOYMENT_MONITORING_GUIDE.md](DEPLOYMENT_MONITORING_GUIDE.md)

### 🚀 專家路徑
1. **全流程自動化** → 使用 `scripts/automation/full_mlops_pipeline.py`
2. **雲端部署** → AWS/GCP/Azure 部署指南
3. **進階監控** → Prometheus + Grafana 設置
4. **CI/CD 整合** → GitHub Actions 工作流

## 🛠️ 專案結構導覽

```
mlops-template/
├── 🏢 domain/              # 核心業務邏輯
│   ├── data/              # 資料管理
│   ├── models/            # 模型開發
│   │   ├── traditional/   # scikit-learn
│   │   ├── deep_learning/ # PyTorch/TensorFlow
│   │   ├── generative/    # LLM/Whisper
│   │   └── specialized/   # NLP/CV/時間序列
│   └── experiments/       # Jupyter 實驗
├── 🔧 application/         # 應用服務
│   ├── training/          # 訓練流水線
│   ├── inference/         # BentoML 推論服務
│   ├── validation/        # 模型驗證
│   └── registry/          # 模型註冊表
├── 🏗️ infrastructure/      # 基礎設施
│   ├── deployment/        # Docker/K8s 配置
│   ├── monitoring/        # Prometheus 監控
│   └── cicd/             # GitHub Actions
├── 🔄 shared/             # 共享工具
│   ├── utils/            # 通用工具
│   └── configs/          # 配置管理
└── 📖 examples/           # 範例和教程
```

## 🎨 使用範例

### 1. 訓練自定義模型

```python
# 在 domain/models/traditional/your_model/ 創建
from sklearn.ensemble import RandomForestClassifier
import joblib

# 訓練您的模型
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 保存到註冊表
joblib.dump(model, 'application/registry/model_registry/your_model.joblib')
```

### 2. 創建 BentoML 服務

```python
# 在 application/inference/services/ 創建
import bentoml
from bentoml.io import JSON

your_model_ref = bentoml.sklearn.get("your_model:latest")

@bentoml.service()
class YourModelService:
    @bentoml.api
    def predict(self, input_data: JSON) -> JSON:
        model = your_model_ref.load_model()
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}
```

### 3. 設置監控

```python
# 使用內建的監控工具
from infrastructure.monitoring.metrics.model_metrics import ModelMetricsCollector

collector = ModelMetricsCollector("your_model", "1.0.0")
collector.record_prediction(features, prediction, confidence, latency)
```

## 🎯 常見使用場景

### 場景 1: 快速原型開發
```bash
# 使用 Jupyter 進行實驗
cd domain/experiments/notebooks
poetry run jupyter lab

# 快速測試模型
poetry run python domain/models/traditional/sklearn/quickstart.py
```

### 場景 2: 生產部署
```bash
# 完整流水線
poetry run python scripts/automation/full_mlops_pipeline.py

# 容器化部署
make containerize
docker run -p 3000:3000 iris_classifier:latest
```

### 場景 3: 持續整合
```bash
# 在 CI/CD 中使用
poetry run python application/validation/model_validation/validate_model.py --threshold 0.95
```

## 📊 監控儀表板

啟動完整監控棧：

```bash
# 方式 1: Docker Compose
docker-compose -f infrastructure/monitoring/docker-compose.yml up -d

# 方式 2: 手動啟動
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3001 (admin/admin)
```

訪問儀表板：
- **服務監控**: http://localhost:3000/get_metrics_summary
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001

## 🔧 故障排除

### 常見問題快速解決

#### 問題 1: Poetry 安裝失敗
```bash
# 解決方案
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

#### 問題 2: GPU 不可用
```bash
# 檢查 NVIDIA 驅動
nvidia-smi

# 安裝 CUDA (如需要)
# 查看官方文檔: https://developer.nvidia.com/cuda-downloads
```

#### 問題 3: 埠號衝突
```bash
# 檢查埠號使用
netstat -tulpn | grep :3000

# 使用不同埠號
poetry run bentoml serve iris_service:IrisClassifier --port 3001
```

#### 問題 4: 模型載入失敗
```bash
# 檢查模型文件
ls -la application/registry/model_registry/

# 重新訓練
poetry run python application/training/pipelines/iris_training_pipeline.py
```

## 📖 延伸閱讀

- 📘 **完整教學**: [MLOPS_COMPLETE_TUTORIAL.md](MLOPS_COMPLETE_TUTORIAL.md)
- 🏗️ **系統架構**: [SYSTEM_DESIGN_RESTRUCTURE.md](SYSTEM_DESIGN_RESTRUCTURE.md)
- 🚀 **部署指南**: [DEPLOYMENT_MONITORING_GUIDE.md](DEPLOYMENT_MONITORING_GUIDE.md)
- 📦 **Poetry 使用**: [POETRY_QUICKSTART.md](POETRY_QUICKSTART.md)
- 🔄 **遷移報告**: [MIGRATION_TO_POETRY.md](MIGRATION_TO_POETRY.md)

## 🤝 貢獻指南

想要貢獻這個專案？

1. 🍴 Fork 這個專案
2. 🌿 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 💾 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 📤 推送到分支 (`git push origin feature/AmazingFeature`)
5. 🔀 開啟 Pull Request

## 📞 支援

需要幫助？

- 📖 查看文檔目錄中的相關指南
- 🐛 提交 Issue 報告問題
- 💬 在 Discussions 中討論

**開始您的 MLOps 之旅吧！** 🎉

---

*最後更新: 2024年*