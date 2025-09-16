# 🚀 MLOps 專案入門指南

## 🎯 快速開始

### 一鍵執行完整流程

```bash
# 推薦：一鍵執行完整流水線 (包含所有修復)
bash scripts/quickstart.sh

# 或分步驟執行：
# 步驟 1: 安裝依賴
make install

# 步驟 2: 檢查 GPU
make checkgpu

# 步驟 3: 訓練模型
poetry run python application/training/pipelines/iris_training_pipeline.py

# 步驟 4: 啟動服務 (無警告版本)
make run

# 或使用 Python 腳本
poetry run python scripts/automation/full_mlops_pipeline.py
```

這個指令將自動執行完整的 MLOps 流水線：
1. ✅ 檢查系統需求和依賴
2. 📦 安裝 Poetry 和所有依賴項
3. 🔧 驗證 GPU 環境和 CUDA 支持
4. 📁 創建必要的專案目錄結構
5. 🏃 訓練 Iris 分類模型 (包含 MLflow 追蹤)
6. ✅ 驗證模型品質和效能指標
7. 🚀 建立 BentoML 服務 (使用最新 API)
8. 🧪 執行功能測試和負載測試
9. 📊 啟動監控指標和健康檢查

**💡 為什麼推薦使用 quickstart.sh？**
- 自動修復常見問題 (domain 目錄、API 兼容性等)
- 使用無警告版本啟動服務
- 完整的錯誤處理和清理機制
- 一步到位，無需手動處理每個步驟

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

# 啟動 BentoML 服務 (無警告)
make run

# 容器化部署
make containerize
docker run -p 3000:3000 iris_classifier:latest
```

### BentoML 服務啟動指南

#### 正確的服務啟動方式
```bash
# 方式 1: 使用 Makefile (推薦)
make run

# 方式 2: 手動啟動 (無警告版本)
cd application/inference/services
PYTHONWARNINGS="ignore" poetry run bentoml serve iris_service.py:svc --reload

# 方式 3: 使用 bentofile.yaml
cd application/inference/services
poetry run bentoml serve . --reload
```

#### 常見問題解決
- **警告訊息**: 使用 `PYTHONWARNINGS="ignore"` 環境變數
- **API 錯誤**: 確保使用 `@bentoml.Service` 而非 `@bentoml.service`
- **模型載入**: 使用 `bentoml.sklearn.get("model_name:latest").to_runner()`

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
poetry run bentoml serve iris_service.py:svc --port 3001 --reload
```

#### 問題 4: BentoML API 兼容性
```bash
# 檢查 BentoML 版本
poetry run bentoml --version

# 常見錯誤修復
# 錯誤: AttributeError: module 'bentoml' has no attribute 'service'
# 解決: 使用 @bentoml.Service 而非 @bentoml.service

# 錯誤: TypeError: Service.__init__() got an unexpected keyword argument 'resources'
# 解決: 將 resources 參數移到 @bentoml.api 裝飾器
```

#### 問題 4: 模型載入失敗
```bash
# 檢查模型文件
ls -la application/registry/model_registry/

# 重新訓練
poetry run python application/training/pipelines/iris_training_pipeline.py
```

## 🔧 Makefile 命令完整參考

### 設置與安裝

#### `make install` - 安裝所有依賴 (推薦)
```bash
make install
```
- 安裝 Poetry 依賴項
- 配置 GPU 支持 (PyTorch, TensorFlow)
- 安裝 OpenAI Whisper
- **適用場景**: 完整開發環境設置

#### `make install-dev` - 開發工具安裝 (最小化)
```bash
make install-dev
```
- 安裝基本開發工具 (black, pylint, pytest, jupyter)
- **適用場景**: 快速開發環境設置

### 開發工作流

#### `make refactor` - 代碼重構 (格式化 + 檢查)
```bash
make refactor
```
- 運行 `make format` 和 `make lint`
- **適用場景**: 代碼質量改進

#### `make format` - 代碼格式化
```bash
make format
```
- 使用 Black 格式化代碼
- **適用場景**: 統一代碼風格

#### `make lint` - 代碼檢查
```bash
make lint
```
- 使用 Pylint 檢查代碼品質
- **適用場景**: 代碼質量檢查

#### `make test` - 運行測試
```bash
make test
```
- 運行 pytest 測試套件
- 生成覆蓋率報告
- **適用場景**: 驗證代碼功能

#### `make clean` - 清理構建文件
```bash
make clean
```
- 刪除 `__pycache__` 目錄
- 清理 `.pyc` 文件
- 移除 `dist/`, `build/` 目錄
- **適用場景**: 清理開發環境

### ML 與 GPU

#### `make checkgpu` - GPU 環境檢查
```bash
make checkgpu
```
- 驗證 PyTorch CUDA 支持
- 檢查 TensorFlow GPU 支持
- **適用場景**: GPU 配置驗證

#### `make train` - 訓練模型
```bash
make train
```
- 運行模型訓練流水線
- **適用場景**: 模型訓練

### 部署與服務

#### `make bento-build` - 構建 BentoML 服務
```bash
make bento-build
```
- **功能**: 根據 `bentofile.yaml` 構建完整的 BentoML 服務包
- **輸入**: 從 BentoML store 載入已訓練的模型
- **輸出**: 生成可部署的 BentoML 服務包
- **適用場景**: 生產環境服務準備

**📋 詳細流程說明：**

1. **讀取配置**: 解析 `bentofile.yaml` 中的服務配置
2. **打包模型**: 從 BentoML store 載入已訓練的模型 (`iris_clf:latest`)
3. **打包代碼**: 包含所有必要的 Python 文件和依賴
4. **創建環境**: 設置 Python 環境和系統依賴
5. **生成服務**: 創建可執行的 BentoML 服務包

**🔗 與其他命令的關係：**
- **前置條件**: 需要先運行 `poetry run python application/training/pipelines/iris_training_pipeline.py`
- **後續步驟**: 可使用 `make containerize` 進一步容器化

#### `make containerize` - 容器化服務
```bash
make containerize
```
- 創建 Docker 容器
- **適用場景**: 生產部署準備

#### `make run` - 啟動本地服務 (無警告)
```bash
make run
```
- 啟動 BentoML 服務 (已包含警告抑制)
- **適用場景**: 本地開發測試

#### `make deploy` - 部署服務
```bash
make deploy
```
- 部署到生產環境 (需配置)
- **適用場景**: 生產部署

### 綜合命令

#### `make all` - 完整流水線
```bash
make all
```
- 運行: install → format → lint → test → checkgpu
- **適用場景**: 完整環境設置和驗證

#### `make help` - 顯示幫助 (默認)
```bash
make help  # 或只輸入 make
```
- 顯示所有可用命令說明
- **適用場景**: 查看命令幫助

## 🚀 完整 MLOps 流程：從模型到 API

### 階段 1：模型訓練與保存
```bash
# 訓練模型並保存到 BentoML store
poetry run python application/training/pipelines/iris_training_pipeline.py
```
**輸出**:
- MLflow 實驗記錄
- 本地模型文件 (`application/registry/model_registry/`)
- BentoML 模型 (`iris_clf:latest`)

### 階段 2：服務開發
```python
# application/inference/services/iris_service.py
iris_model_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service(
    name="iris_classifier",
    runners=[iris_model_runner],
)

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> np.ndarray:
    return await iris_model_runner.predict.async_run(input_series)
```

### 階段 3：服務構建 (`bento build`)
```bash
# 根據 bentofile.yaml 構建服務包
make bento-build
```
**輸入**:
- `bentofile.yaml` (服務配置)
- BentoML store 中的模型
- 服務代碼文件

**輸出**:
- BentoML 服務包 (包含模型、代碼、依賴、環境配置)

### 階段 4：容器化部署
```bash
# 可選：創建 Docker 容器
make containerize
```

### 階段 5：服務啟動
```bash
# 啟動生產服務
make run

# 或手動啟動
PYTHONWARNINGS="ignore" poetry run bentoml serve iris_service.py:svc --reload
```

### 📊 數據流圖
```
原始數據 → 模型訓練 → BentoML Store → 服務構建 → API 服務
    ↓         ↓         ↓            ↓         ↓
  CSV文件 → Scikit-learn → iris_clf:latest → bento build → http://localhost:3000
```

### 🔗 關鍵文件
- **訓練**: `application/training/pipelines/iris_training_pipeline.py`
- **模型載入**: `application/inference/services/iris_service.py`
- **服務配置**: `application/inference/services/bentofile.yaml`
- **構建命令**: `make bento-build`

**💡 `bento build` 的作用**:
`bento build` 是將訓練好的模型和服務代碼打包成可部署的生產服務包的關鍵步驟。它解決了從開發環境到生產環境的遷移問題，確保服務可以在任何環境中一致運行。

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