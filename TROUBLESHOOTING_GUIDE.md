# 🔧 MLOps 故障排除指南

## 🎯 概述

本文檔收集了在 MLOps 系統開發和部署過程中遇到的常見問題及其解決方案，特別針對 BentoML 服務的各種問題。

## 📋 目錄

- [BentoML 服務問題](#bentoml-服務問題)
- [環境設置問題](#環境設置問題)
- [模型訓練問題](#模型訓練問題)
- [部署問題](#部署問題)
- [監控問題](#監控問題)

---

## 🚀 BentoML 服務問題

### 問題 1: API 兼容性錯誤

#### 錯誤訊息
```python
AttributeError: module 'bentoml' has no attribute 'service'. Did you mean: 'Service'?
```

#### 原因分析
BentoML 在版本更新中將 `@bentoml.service` 改為 `@bentoml.Service`（大寫 S）。

#### 解決方案
```python
# ❌ 錯誤的寫法 (舊版本)
@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 20},
)
class MyService:
    pass

# ✅ 正確的寫法 (新版本)
@bentoml.Service(
    name="my_service",
    runners=[model_runner],
)
class MyService:
    pass
```

### 問題 2: 服務參數錯誤

#### 錯誤訊息
```python
TypeError: Service.__init__() got an unexpected keyword argument 'resources'
```

#### 原因分析
新版本 BentoML 中，`resources` 和 `traffic` 參數不能直接放在 `@bentoml.Service` 裝飾器中。

#### 解決方案
```python
# ❌ 錯誤的寫法
@bentoml.Service(
    name="my_service",
    resources={"cpu": "2"},
    traffic={"timeout": 20},
)
class MyService:
    pass

# ✅ 正確的寫法
@bentoml.Service(
    name="my_service",
    runners=[model_runner],
)
class MyService:
    @bentoml.api(
        resources={"cpu": "2"},
        traffic={"timeout": 20},
    )
    def predict(self, input_data):
        pass
```

### 問題 3: 服務實例名稱錯誤

#### 錯誤訊息
```python
Attribute "IrisClassifier" not found in module "iris_service".
```

#### 原因分析
服務實例名稱與實際定義的變數名稱不符。

#### 解決方案
```bash
# ❌ 錯誤的啟動命令
poetry run bentoml serve iris_service.py:IrisClassifier --reload

# ✅ 正確的啟動命令 (檢查實際的服務實例名稱)
poetry run bentoml serve iris_service.py:svc --reload
```

### 問題 4: 模型載入方式錯誤

#### 錯誤訊息
```python
AttributeError: 'Model' object has no attribute 'predict'
```

#### 原因分析
新版本 BentoML 需要使用 `runner` 模式來載入和使用模型。

#### 解決方案
```python
# ❌ 錯誤的寫法 (舊版本)
iris_model_ref = bentoml.sklearn.get("iris_clf:latest")

class MyService:
    def __init__(self):
        self.model = iris_model_ref.load_model()
    
    def predict(self, input_data):
        return self.model.predict(input_data)

# ✅ 正確的寫法 (新版本)
iris_model_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

@bentoml.Service(
    name="iris_service",
    runners=[iris_model_runner],
)
class MyService:
    def predict(self, input_data):
        return iris_model_runner.predict.run(input_data)
```

### 問題 5: 警告訊息處理

#### 錯誤訊息
```
UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
```

#### 原因分析
第三方依賴 `fs` 使用了已棄用的 `pkg_resources` API。

#### 解決方案
```bash
# 方式 1: 使用環境變數 (推薦)
PYTHONWARNINGS="ignore" poetry run bentoml serve iris_service.py:svc --reload

# 方式 2: 在 Makefile 中設定
run:
	@echo "🚀 Starting BentoML service..."
	cd application/inference/services && PYTHONWARNINGS="ignore" poetry run bentoml serve iris_service.py:svc --reload
```

---

## 🛠️ 環境設置問題

### 問題 1: Poetry 安裝失敗

#### 錯誤訊息
```
make: *** [Makefile:6: install] Error 1
```

#### 原因分析
專案結構中缺少必要的 `__init__.py` 檔案。

#### 解決方案
```bash
# 創建缺少的目錄和檔案
mkdir -p domain
touch domain/__init__.py

# 重新安裝
make install
```

### 問題 2: GPU 不可用

#### 錯誤訊息
```
CUDA not available
```

#### 解決方案
```bash
# 檢查 NVIDIA 驅動
nvidia-smi

# 檢查 CUDA 安裝
poetry run python -c "import torch; print(torch.cuda.is_available())"

# 重新安裝 CUDA 版本的 PyTorch
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 問題 3: 埠號衝突

#### 錯誤訊息
```
Address already in use: Port 3000
```

#### 解決方案
```bash
# 檢查埠號使用
netstat -tulpn | grep :3000

# 使用不同埠號
poetry run bentoml serve iris_service.py:svc --port 3001 --reload

# 或終止佔用埠號的程序
sudo kill -9 $(lsof -t -i:3000)
```

---

## 🎯 模型訓練問題

### 問題 1: MLflow 連接失敗

#### 錯誤訊息
```
NoCredentialsError: Unable to locate credentials
```

#### 解決方案
```python
# 設置 MLflow 追蹤 URI 和 MinIO 憑證
import os
import mlflow

# 設置追蹤 URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 設置 MinIO 憑證
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9010"
```

### 問題 2: MinIO Bucket 不存在

#### 錯誤訊息
```
NoSuchBucket: The specified bucket does not exist
```

#### 解決方案
```bash
# 創建 MLflow artifacts bucket
docker run --rm minio/mc alias set localminio http://127.0.0.1:9010 minioadmin minioadmin
docker run --rm minio/mc mb localminio/mlflow-artifacts
```

### 問題 3: 模型保存失敗

#### 錯誤訊息
```
PermissionError: [Errno 13] Permission denied
```

#### 解決方案
```bash
# 檢查目錄權限
ls -la application/registry/model_registry/

# 創建目錄並設置權限
mkdir -p application/registry/model_registry
chmod 755 application/registry/model_registry
```

---

## 🚀 部署問題

### 問題 1: Docker 映像構建失敗

#### 錯誤訊息
```
ERROR: failed to solve: failed to read dockerfile
```

#### 解決方案
```bash
# 檢查 Dockerfile 是否存在
ls -la infrastructure/deployment/containers/docker/Dockerfile

# 使用正確的路徑構建
cd infrastructure/deployment/containers/docker/
docker build -t my-model .
```

### 問題 2: 容器啟動失敗

#### 錯誤訊息
```
Container exited with code 1
```

#### 解決方案
```bash
# 查看容器日誌
docker logs <container_id>

# 檢查模型檔案是否存在
docker exec -it <container_id> ls -la /app/models/

# 重新構建並啟動
docker build --no-cache -t my-model .
docker run -p 3000:3000 my-model
```

---

## 📊 監控問題

### 問題 1: Prometheus 指標缺失

#### 錯誤訊息
```
No metrics found for target
```

#### 解決方案
```bash
# 檢查 Prometheus 配置
curl http://localhost:9090/targets

# 檢查指標端點
curl http://localhost:8001/metrics

# 重啟監控服務
docker-compose restart prometheus grafana
```

### 問題 2: Grafana 儀表板無法載入

#### 錯誤訊息
```
Failed to load dashboard
```

#### 解決方案
```bash
# 檢查 Grafana 日誌
docker logs grafana

# 檢查資料源配置
curl http://localhost:3001/api/datasources

# 重新配置資料源
# 訪問 http://localhost:3001 並添加 Prometheus 資料源
```

---

## 🎯 最佳實踐建議

### 1. 版本管理
- 始終使用 `poetry.lock` 確保環境一致性
- 定期更新依賴項：`poetry update`
- 使用語義化版本標記：`^1.2.3`

### 2. 錯誤處理
- 在服務中添加適當的錯誤處理
- 使用日誌記錄重要的操作和錯誤
- 實施健康檢查端點

### 3. 測試策略
- 編寫單元測試和整合測試
- 使用負載測試驗證性能
- 實施自動化測試流程

### 4. 監控和觀察
- 設置關鍵指標監控
- 實施告警機制
- 定期檢查系統健康狀態

---

### 自動修復腳本

如果遇到多個問題，建議使用自動修復腳本：

```bash
# 一鍵執行完整流水線 (包含所有修復)
bash scripts/quickstart.sh

# 或只安裝依賴 (修復常見設置問題)
bash scripts/quickstart.sh --install-only

# 或使用專門的 Poetry 設置腳本
bash scripts/setup/setup_poetry.sh
```

**💡 自動修復腳本能解決的問題：**
- ✅ domain 目錄缺失問題
- ✅ Poetry 依賴安裝問題
- ✅ GPU 環境配置問題
- ✅ BentoML API 兼容性問題
- ✅ 服務啟動警告問題

### Makefile 命令故障排除

```bash
# 檢查所有可用命令
make help

# 檢查 Poetry 環境狀態
poetry env info

# 重新安裝所有依賴
make clean && make install

# 檢查 GPU 配置
make checkgpu

# 測試服務啟動
make run
```

**常見 Makefile 錯誤解決：**

#### `make: command not found`
```bash
# 安裝 make (Ubuntu/Debian)
sudo apt-get install make

# 安裝 make (macOS with Homebrew)
brew install make
```

#### `poetry: command not found`
```bash
# 重新安裝 Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

#### `make install` 失敗
```bash
# 檢查 Poetry 配置
poetry config --list

# 清理 Poetry 緩存
poetry cache clear --all pypi

# 重新初始化
rm -rf ~/.cache/pypoetry
make install
```

#### `make run` 服務啟動失敗
```bash
# 檢查 BentoML 模型
poetry run bentoml models list

# 檢查服務狀態
poetry run bentoml list

# 查看日誌
tail -f bentoml_service.log
```

## 📞 獲取幫助

如果遇到本文檔未涵蓋的問題：

1. **首先嘗試自動修復腳本**：`bash scripts/quickstart.sh`
2. 檢查專案的 [README.md](README.md)
3. 查看 [MLOPS_COMPLETE_TUTORIAL.md](MLOPS_COMPLETE_TUTORIAL.md)
4. 參考 [DEPLOYMENT_MONITORING_GUIDE.md](DEPLOYMENT_MONITORING_GUIDE.md)
5. 提交 Issue 到專案儲存庫

**記住：每次遇到問題都是一次學習的機會！** 🎓
