# Poetry 快速入門指南

## 🚀 為什麼使用 Poetry？

Poetry 解決了 Python 依賴管理的痛點：
- **精確的版本鎖定** - 避免「在我電腦上可以跑」的問題
- **依賴解析** - 自動處理依賴衝突
- **虛擬環境管理** - 自動創建和管理虛擬環境
- **構建與發布** - 標準化的打包流程

## 📦 安裝 Poetry

```bash
# 官方安裝方式
curl -sSL https://install.python-poetry.org | python3 -

# 或使用 pip（不推薦）
pip install poetry
```

## 🎯 快速開始

### 1. 開發環境設置

```bash
# 複製專案後，安裝所有依賴
make install

# 或手動使用 Poetry
poetry install --with dev --extras all

# 激活虛擬環境
poetry shell

# 檢查 GPU 設置
make checkgpu
```

### 2. 日常開發工作流

```bash
# 格式化和檢查代碼
make refactor

# 運行測試
make test

# 添加新依賴
poetry add numpy pandas

# 添加開發依賴
poetry add --group dev pytest black

# 查看已安裝的包
poetry show

# 更新依賴
poetry update
```

### 3. 運行 ML 範例

```bash
# 激活環境
poetry shell

# 運行 scikit-learn 範例
cd domain/models/traditional/sklearn
jupyter lab iris_classifier.ipynb

# 運行 Hugging Face 範例
poetry run python domain/models/generative/llm/zero_shot_classification.py classify

# 運行 BentoML 服務
make run
```

## 📋 Poetry 常用命令

### 依賴管理
```bash
poetry add <package>              # 添加生產依賴
poetry add --group dev <package>  # 添加開發依賴
poetry remove <package>           # 移除依賴
poetry show                       # 顯示已安裝的包
poetry show --tree               # 顯示依賴樹
poetry update                     # 更新所有依賴
poetry update <package>           # 更新特定包
```

### 虛擬環境
```bash
poetry shell                     # 激活虛擬環境
poetry run <command>             # 在虛擬環境中運行命令
poetry env info                  # 顯示環境資訊
poetry env list                  # 列出所有環境
poetry env remove <env-name>     # 刪除環境
```

### 構建與發布
```bash
poetry build                     # 構建 wheel 和 sdist
poetry publish                   # 發布到 PyPI
poetry version <version>         # 更新版本號
```

## 🔧 配置檔案說明

### pyproject.toml 關鍵區塊

```toml
[tool.poetry.dependencies]
# 生產依賴
python = "^3.9"
torch = {version = "^1.12.0", optional = true}

[tool.poetry.group.dev.dependencies]
# 開發依賴（不會安裝到生產環境）
pytest = "^7.1.3"
black = "^22.3.0"

[tool.poetry.extras]
# 可選功能組
torch = ["torch", "torchvision"]
all = ["torch", "torchvision", "tensorflow"]
```

## 🎯 專案特定配置

### GPU 依賴安裝
```bash
# PyTorch with CUDA
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TensorFlow with CUDA
poetry run pip install tensorflow[and-cuda]

# 或使用 Makefile
make install
```

### 開發環境激活
```bash
# 方式 1: 激活 shell
poetry shell
python domain/models/deep_learning/pytorch/quickstart_pytorch.py

# 方式 2: 直接運行
poetry run python domain/models/deep_learning/pytorch/quickstart_pytorch.py

# 方式 3: 使用 Makefile
make checkgpu
```

## 🐳 Dev Container 整合

Dev Container 已配置為自動：
1. 安裝 Poetry
2. 安裝專案依賴
3. 配置 VS Code Python 解釋器
4. 設置 GPU 支援

```bash
# Dev Container 啟動後自動執行
bash scripts/setup/setup_poetry.sh
```

## 🆚 Poetry vs pip 對比

| 功能 | Poetry | pip + virtualenv |
|------|---------|------------------|
| 依賴解析 | ✅ 自動 | ❌ 手動處理衝突 |
| 鎖定檔案 | ✅ poetry.lock | ❌ 無標準格式 |
| 虛擬環境 | ✅ 自動管理 | ❌ 手動創建 |
| 版本管理 | ✅ 語義化版本 | ❌ 基本支援 |
| 構建打包 | ✅ 內建支援 | ❌ 需要額外工具 |

## 🚨 常見問題

### Q: Poetry 安裝很慢？
A: 使用國內鏡像源
```bash
poetry config repositories.pypi https://mirrors.aliyun.com/pypi/simple/
poetry config pypi-token.pypi your-token
```

### Q: 如何在 Docker 中使用？
A: 參考 `.devcontainer/Dockerfile`
```dockerfile
RUN curl -sSL https://install.python-poetry.org | python3 -
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-dev
```

### Q: 需要 requirements.txt 文件嗎？
A: 不需要！Poetry 完全取代了 requirements.txt
```bash
# 唯一推薦的安裝方式
make install

# 或直接使用 Poetry
poetry install --with dev --extras all

# 最小化開發工具安裝（不推薦）
make install-dev
```

## 🎉 最佳實踐

1. **總是使用 poetry.lock** - 確保環境一致性
2. **區分依賴類型** - 生產 vs 開發 vs 可選
3. **定期更新依賴** - `poetry update`
4. **使用語義化版本** - `^1.2.3` 允許小版本更新
5. **提交 poetry.lock** - 團隊協作必需

開始您的 MLOps 開發之旅！🚀