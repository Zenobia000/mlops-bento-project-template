# MLOps 專案架構分析文件

本文檔旨在以系統分析 (SA) 與系統設計 (SD) 的角度，對本 MLOps 專案範本進行架構分析、功能拆解與組件相依性說明。

## 1. 專案總體架構 (Overall Architecture)

本專案是一個標準化的機器學習運維 (MLOps) 系統範本，旨在提供一個從模型開發、訓練、打包、部署到自動化的完整解決方案。

其核心理念是將機器學習模型視為一個軟體工程產物，透過 CI/CD、容器化和服務化，實現高效、可靠且可重複的部署流程。

主要技術棧包括：
- **程式語言**: Python
- **模型服務化**: BentoML
- **機器學習框架**: Hugging Face Transformers, PyTorch, TensorFlow
- **容器化**: Docker
- **CI/CD**: GitHub Actions
- **開發環境**: VS Code Dev Containers

### 架構流程圖

```
+-------------------------+      +------------------+      +--------------------+
|   模型開發與實驗        |----->|   模型訓練       |----->|  模型打包與服務化  |
| (hugging-face/, .ipynb) |      | (train.py)       |      | (BentoML service.py) |
+-------------------------+      +------------------+      +--------------------+
                                                                   |
                                                                   v
+-------------------------+      +--------------------+      +--------------------+
|      自動化部署         |<-----|  CI/CD 工作流      |<-----|     容器化         |
|   (e.g., K8s, Cloud)    |      | (GitHub Actions)   |      |   (Dockerfile)     |
+-------------------------+      +--------------------+      +--------------------+
```

---

## 2. 功能拆解 (Functional Decomposition)

從 SA/SD 角度，我們可將此專案拆解為以下幾個核心功能組件：

### 2.1. 模型開發與實驗 (Model Development & Experimentation)
- **目的**: 提供資料科學家進行模型探索、演算法開發與實驗的環境。
- **相關檔案**:
    - `hugging-face/`: 包含使用 Hugging Face 模型（如 Whisper）的範例腳本，用於下載、微調和執行推論。
    - `examples/quickstart/iris_classifier.ipynb`: Jupyter Notebook 提供互動式的開發體驗。
    - `main.py`: 專案主入口，可能用於觸發特定流程或進行快速測試。
    - `test_main.py`: 對核心邏輯的單元測試。

### 2.2. 模型服務化 (Model Serving)
- **目的**: 將訓練好的模型封裝成一個標準化的 RESTful API 服務，使其易於被其他應用程式呼叫。
- **相關檔案**:
    - `bentoml/`: 存放 BentoML 相關範例與設定。
    - `examples/quickstart/service.py`: 定義 BentoML 服務，包含 API 端點、輸入/輸出格式以及模型推論邏輯。
    - `bentofile.yaml`: BentoML 的設定檔，定義服務的依賴項、要包含的模型以及 Python 套件。
    - `examples/quickstart/query.py`, `query.sh`: 用於測試已啟動模型服務的客戶端腳本。

### 2.3. 容器化 (Containerization)
- **目的**: 將應用程式（特別是模型服務）及其所有依賴項打包成一個獨立、可移植的 Docker 映像檔。
- **相關檔案**:
    - `Dockerfile`: 用於建置包含 BentoML 服務的正式產品級 Docker 映像檔。
    - `.devcontainer/Dockerfile`: 用於建立一個標準化、一致的開發環境，供 VS Code Dev Containers 使用。

### 2.4. 自動化與 CI/CD (Automation & CI/CD)
- **目的**: 自動化測試、建置和部署流程，確保程式碼品質並加速交付週期。
- **相關檔案**:
    - `.github/workflows/`: 存放 GitHub Actions 的工作流設定。
        - `static.yml`: 執行靜態程式碼分析（如 Linting）。
        - `cicd.yml`: 核心的持續整合與持續部署流程，可能包含：執行測試、建置 BentoML 服務、建置 Docker 映像檔、推送到容器儲存庫。
        - `docker-image.yml`: 專門用於建置和發布 Docker 映像檔的工作流。
    - `Makefile`: 提供一組標準化的命令捷徑（如 `make test`, `make build`），簡化開發與 CI/CD 腳本的複雜度。

### 2.5. 輔助工具與函式庫 (Utilities & Libraries)
- **目的**: 提供可重用的函式庫和輔助腳本，支援主要功能。
- **相關檔案**:
    - `mylib/`: 自定義的 Python 函式庫（例如 `calculator.py`），可擴充專案的業務邏輯。
    - `utils/`: 包含各種輔助腳本，如環境驗證 (`verify_pytorch.py`, `verify_tf.py`)、音訊處理 (`transcribe-whisper.sh`) 等。
    - `requirements.txt`, `tf-requirements.txt`: 定義專案的 Python 套件依賴。

---

## 3. 組件相依性 (Component Dependencies)

- **模型服務化** (`bentoml/service.py`) 依賴於 **模型開發** 階段產出的模型檔案（例如 `*.pt`, `*.h5` 或 Hugging Face 模型）。它也依賴 `bentofile.yaml` 來定義其環境。

- **容器化** (`Dockerfile`) 依賴於 **模型服務化** 的成果。Dockerfile 通常會從 BentoML 的本地儲存庫中讀取已建置的 "Bento" 來建立最終的服務映像檔。

- **CI/CD 工作流** (`.github/workflows/*.yml`) 是整個架構的協調者：
    - 它會觸發測試 (`test_main.py`)。
    - 它使用 `bentoml` CLI 和 `bentofile.yaml` 來建置模型服務。
    - 它使用 `Dockerfile` 來建置 Docker 映像檔。
    - 它可能會將建置好的映像檔推送到 Docker Hub 或其他容器儲存庫。

- **模型開發** (`hugging-face/`, `main.py`) 依賴 `mylib` 和 `utils` 中的輔助函式，以及 `requirements.txt` 中定義的函式庫。

- **開發環境** (`.devcontainer/`) 依賴 `requirements.txt` 來預先安裝開發所需的套件，為開發者提供一個與正式環境相似的無縫體驗。

---

## 4. 建議工作流程 (Recommended Workflow)

1.  **環境設定**: 使用 VS Code 和 Dev Container 功能，基於 `.devcontainer/` 設定檔一鍵啟動開發環境。
2.  **模型開發**: 在 `hugging-face/` 或新的資料夾中開發模型訓練腳本和實驗。
3.  **函式庫開發**: 在 `mylib/` 中新增或修改通用的商業邏輯。
4.  **服務定義**: 在 `bentoml/` 或 `examples/` 路徑下，建立 `service.py` 和 `bentofile.yaml` 來定義模型如何被服務化。
5.  **本地測試**:
    - 執行 `pytest` 進行單元測試。
    - 執行 `bentoml serve` 在本地啟動模型服務。
    - 使用 `query.py` 或 `curl` 測試服務端點。
6.  **提交與推送**: 將程式碼提交到 Git，並推送到遠端儲存庫。
7.  **自動化流程**: GitHub Actions 將自動被觸發，執行測試、程式碼掃描、建置 Docker 映像檔，並可能將其部署到預備或正式環境。

## 5. 結論

本專案範本提供了一個結構清晰、高度自動化的 MLOps 基礎架構。透過將模型開發、服務化、容器化和 CI/CD 流程的解耦與整合，它能有效地幫助團隊標準化機器學習的生命週期管理，提升開發效率與部署品質。
