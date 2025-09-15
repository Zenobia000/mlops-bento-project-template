# Dev Container 開發環境說明

本文檔旨在說明 `.devcontainer/` 目錄下定義的開發環境的狀態、配置與初始化流程。

## 總覽

本專案使用 VS Code Dev Containers (開發容器) 來建立一個標準化、可重現的開發環境。此環境基於 Docker 容器技術，並預先配置了 GPU 支援、Python 虛擬環境以及多種開發工具，確保所有開發者都擁有一致的開發體驗。

核心設定檔案包含：
- `.devcontainer/devcontainer.json`: Dev Container 的主要設定檔，定義了容器的建置方式、VS Code 整合、功能擴充等。
- `.devcontainer/Dockerfile`: 用於建置 Docker 映像的檔案，定義了基礎系統、安裝的軟體套件與環境設定。
- `setup.sh`: 在容器建立後執行的初始化腳本。

---

## 環境狀態與配置

### 1. 基礎映像與系統

- **基礎映像**: `mcr.microsoft.com/vscode/devcontainers/universal:2-focal`
  - 這是一個由 Microsoft 提供的通用開發映像，基於 **Ubuntu 20.04 (Focal Fossa)**。
- **系統套件**:
  - `ffmpeg`: 用於處理音訊和視訊。
  - `python3.8-venv`: 用於建立 Python 虛擬環境。
  - `gcc`: C 語言編譯器。
  - `pciutils`: 用於查詢 PCI 設備 (例如 GPU)。

### 2. GPU 支援 (NVIDIA CUDA)

為了支援機器學習與深度學習開發，此環境完整整合了 NVIDIA GPU：

- **CUDA & cuDNN**:
  - `devcontainer.json` 中透過 `"ghcr.io/devcontainers/features/nvidia-cuda:1"` 功能自動安裝 NVIDIA CUDA Toolkit 和 cuDNN，無需手動設定。
- **NVIDIA Container Toolkit**:
  - `Dockerfile` 中會自動安裝 `nvidia-docker2`，這是讓 Docker 容器能夠存取主機 GPU 資源的關鍵工具。
- **容器執行參數**:
  - `devcontainer.json` 的 `runArgs` 包含了 `--privileged` 等參數，以確保容器有足夠權限與硬體互動。
  - **注意**: `"--gpus", "all"` 參數目前被註解，若要啟用 GPU，需要將其取消註解。

### 3. Python 環境

- **虛擬環境**:
  - `Dockerfile` 為使用者 `codespace` 在路徑 `/home/codespace/venv` 建立了一個獨立的 Python 3.8 虛擬環境。
- **套件安裝**:
  - 容器在建置過程中會自動將根目錄的 `requirements.txt` 複製進來，並使用 `pip` 將所有依賴套件安裝到此虛擬環境中。

### 4. VS Code 整合

- **預設設定**: `devcontainer.json` 中的 `customizations.vscode.settings` 預設了以下重要設定：
  - **Python 解譯器**: 自動指向虛擬環境中的 Python (`/home/codespace/venv/bin/python`)。
  - **Linter 與 Formatter**: 預設啟用 `pylint`，並設定了 `black`, `flake8` 等工具的路徑。
- **自動安裝的擴充套件**: `extensions` 列表中定義了多個實用的 VS Code 擴充套件，會在進入容器時自動安裝，例如：
  - `ms-python.python` & `ms-python.vscode-pylance`: 提供 Python 語言支援與 IntelliSense。
  - `ms-toolsai.jupyter`: 支援 Jupyter Notebook。
  - `ms-azuretools.vscode-docker`: Docker 整合工具。
  - `GitHub.copilot-nightly`: AI 程式碼輔助工具。

---

## 初始化流程

當開發容器成功建立後，會執行以下初始化流程：

1.  **執行 `postCreateCommand`**:
    - `devcontainer.json` 中的 `postCreateCommand` 屬性指定了容器啟動後要執行的命令：`bash setup.sh`。
2.  **執行 `setup.sh` 腳本**:
    - 此腳本會執行以下操作：
      - `source /home/codespace/venv/bin/activate`: 啟用 Python 虛擬環境，讓後續的終端機能直接使用此環境中的 Python 與套件。
      - `echo '...' >> ~/.bashrc`: 將啟用虛擬環境的指令附加到 `~/.bashrc` 檔案中。這確保了每次開啟新的終端機時，都會自動載入並啟用 Python 虛擬環境，無需手動操作。

總結來說，當您進入這個開發環境時，您會直接處在一個已啟用 GPU 支援、安裝好所有 Python 依賴並與 VS Code 深度整合的環境中，可以立即開始開發工作。
