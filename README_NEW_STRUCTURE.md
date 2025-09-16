# MLOps 系統架構 - 新資料夾結構

## 🎯 設計理念

本專案採用**領域驅動設計 (DDD)** 和**分層架構**原則，將 MLOps 系統重新組織為語義化、模組化的結構。

## 📁 資料夾結構概覽

```
mlops-template/
├── 🏢 domain/              # 領域層 - 核心業務邏輯
│   ├── data/              # 資料領域
│   ├── models/            # 模型領域
│   └── experiments/       # 實驗領域
├── 🔧 application/         # 應用層 - 應用服務
│   ├── training/          # 訓練服務
│   ├── inference/         # 推論服務
│   ├── validation/        # 驗證服務
│   └── registry/          # 註冊表服務
├── 🏗️ infrastructure/      # 基礎設施層
│   ├── deployment/        # 部署基礎設施
│   ├── monitoring/        # 監控基礎設施
│   ├── cicd/             # CI/CD 基礎設施
│   └── security/         # 安全基礎設施
├── 🌐 platform/           # 平台服務層
│   ├── api/              # API 閘道
│   ├── auth/             # 認證授權
│   ├── storage/          # 儲存服務
│   └── messaging/        # 訊息傳遞
├── 🔄 shared/             # 共享層
│   ├── utils/            # 工具程式庫
│   ├── configs/          # 配置管理
│   ├── schemas/          # 資料模式
│   └── constants/        # 常數定義
├── 🧪 tests/              # 測試層
├── 📚 docs/               # 文檔
├── 📖 examples/           # 範例
└── 🔨 scripts/            # 腳本工具
```

## 🎨 各層職責說明

### 🏢 Domain Layer (領域層)
**職責**: 封裝核心業務邏輯和規則

- **data/**: 資料相關的業務邏輯
  - `sources/`: 資料源管理 (raw, processed, external)
  - `pipelines/`: 資料管道邏輯
  - `quality/`: 資料品質檢查

- **models/**: 機器學習模型
  - `traditional/sklearn/`: scikit-learn 模型
  - `deep_learning/pytorch/`: PyTorch 深度學習模型
  - `deep_learning/tensorflow/`: TensorFlow 模型
  - `generative/llm/`: 大語言模型 (Hugging Face)
  - `specialized/nlp/`: 專業 NLP 模型

- **experiments/**: 實驗管理
  - `notebooks/`: Jupyter Notebooks
  - `research/`: 研究型實驗
  - `benchmarks/`: 基準測試

### 🔧 Application Layer (應用層)
**職責**: 組織和協調業務邏輯，提供應用服務

- **training/**: 訓練相關服務
- **inference/**: 推論服務 (包含原 BentoML 服務)
- **validation/**: 模型和資料驗證
- **registry/**: 模型註冊表和製品管理

### 🏗️ Infrastructure Layer (基礎設施層)
**職責**: 提供技術基礎設施支援

- **deployment/**: 部署基礎設施
  - `containers/docker/`: Docker 容器化
  - `containers/kubernetes/`: K8s 編排
  - `cloud/`: 雲端部署 (AWS, GCP, Azure)

- **cicd/**: CI/CD 基礎設施
  - `github_actions/`: GitHub Actions 工作流

### 🔄 Shared Layer (共享層)
**職責**: 提供跨層共享的工具和配置

- **utils/**: 工具程式庫
  - `gpu_utils/`: GPU 相關工具
  - `common/`: 通用工具
  - `data_utils/`: 資料處理工具

## 🚀 使用指南

### 開發工作流程

1. **模型開發**: 在 `domain/models/` 中選擇合適的框架目錄
2. **實驗管理**: 使用 `domain/experiments/` 進行探索性分析
3. **服務開發**: 在 `application/inference/` 中創建推論服務
4. **部署配置**: 在 `infrastructure/deployment/` 中配置部署選項

### 範例使用

```bash
# 運行 GPU 驗證
python shared/utils/gpu_utils/verify_cuda_pytorch.py

# 啟動 scikit-learn 範例
cd domain/models/traditional/sklearn/iris_classifier/
jupyter notebook iris_classifier.ipynb

# 運行 Hugging Face 範例
python domain/models/generative/llm/zero_shot_classification.py classify

# 啟動 BentoML 推論服務
cd application/inference/services/
bentoml serve service.py:IrisClassifier --reload

# 構建 Docker 容器
cd infrastructure/deployment/containers/docker/
docker build -t my-model .
```

## 📊 對比舊結構

| 舊結構 (數字編號) | 新結構 (語義化) | 優勢 |
|------------------|----------------|------|
| `03-models/` | `domain/models/` | 直觀表達模型領域 |
| `06-packaging/` | `application/inference/` | 明確表達推論應用 |
| `09-mlops/` | `infrastructure/cicd/` | 清晰的基礎設施職責 |
| `10-utils/` | `shared/utils/` | 表明跨層共享特性 |

## ✨ 新結構優勢

1. **語義化命名** - 目錄名稱直接反映功能
2. **符合架構原則** - 清晰的分層和職責分離
3. **易於導航** - 開發者無需記憶數字編號
4. **高可擴展性** - 新功能有明確的歸屬位置
5. **模組化設計** - 高內聚低耦合

## 🎯 下一步

1. **豐富領域模型** - 在各框架目錄中添加更多範例
2. **完善應用服務** - 實現完整的訓練和推論管道
3. **強化基礎設施** - 添加監控、安全等功能
4. **文檔完善** - 為各層創建詳細使用指南

---

## 📖 延伸閱讀

- 📘 **完整教學**: [MLOPS_COMPLETE_TUTORIAL.md](MLOPS_COMPLETE_TUTORIAL.md)
- 🏗️ **系統架構**: [SYSTEM_DESIGN_RESTRUCTURE.md](SYSTEM_DESIGN_RESTRUCTURE.md)
- 🚀 **部署指南**: [DEPLOYMENT_MONITORING_GUIDE.md](DEPLOYMENT_MONITORING_GUIDE.md)
- 📦 **Poetry 使用**: [POETRY_QUICKSTART.md](POETRY_QUICKSTART.md)
- 🔧 **故障排除**: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)

---

**這是一個符合現代軟體工程最佳實踐的 MLOps 系統架構！** 🚀