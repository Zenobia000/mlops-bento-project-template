# MLOps 專案入門標準作業流程 (SOP)

歡迎來到 MLOps 的世界！本文件將以大學教科書的模式，從第一性原理出發，採用結構化思維，引導你這位初學者從 0 到 1 掌握這個專案的操作與核心概念。

---

## 第零章：心態與基礎 (Mindset & Basics)

### 1.1 第一性原理：什麼是 MLOps？

忘掉所有複雜的定義，回到根本。

- **傳統軟體開發**：寫程式碼 -> 測試 -> 發布。目標是交付一個**功能 (Function)**。
- **傳統機器學習**：找資料 -> 訓練模型 -> 評估。目標是產出一個**模型 (Model)**。

**MLOps 的第一性原理**：**將「模型」視為「軟體」，用軟體工程的方法來管理模型的整個生命週期。**

這意味著我們需要一個系統化的流程來處理模型的**開發、測試、部署、監控和迭代**，使其變得像現代軟體一樣可靠、可擴展且高度自動化。本專案就是這樣一個流程的具體實現。

### 1.2 麥肯錫結構化思維：如何學習本專案？

我們將採用「由點到線，再到面」的結構化學習路徑：

1.  **點 (The Dots)**：先了解最小的功能單元，例如 `src/train.py` (訓練腳本) 和 `src/service.py` (服務定義檔)。
2.  **線 (The Lines)**：將這些點串聯起來，使用 `Makefile` 手動執行一次從「訓練」到「本地服務」的完整流程。
3.  **面 (The Plane)**：理解整個專案如何透過 GitHub Actions (`cicd.yml`) 將這條線自動化，形成一個完整的 MLOps 作業平面。

---

## 第一章：環境建立 (Environment Setup)

**原理**：**環境一致性是 MLOps 的基石。** 為了避免「在我的電腦上可以跑」的窘境，我們使用 Docker 和 VS Code Dev Container 建立一個可複製、隔離的開發環境。

### 步驟 1.1：安裝必要工具

請確保你的電腦已安裝：
1.  **Git**
2.  **Docker Desktop**
3.  **Visual Studio Code**
4.  **VS Code 擴充功能: Dev Containers** (`ms-vscode-remote.remote-containers`)

### 步驟 1.2：啟動開發環境

1.  打開 VS Code。
2.  點擊左下角的綠色 `><` 圖示，選擇 **"Dev Containers: Open Folder in Container..."**。
3.  選擇本專案的根目錄。
4.  VS Code 會讀取 `.devcontainer/` 設定，自動建立一個包含所有工具的 Docker 容器。
5.  成功後，VS Code 的終端機已在容器內部，所有依賴項都已根據 `requirements.txt` 安裝完畢。

**恭喜！你已擁有一個完美的、與世隔絕的開發環境。**

---

## 第二章：核心應用解析 (Dissecting the Core Application)

**原理**：**關注點分離。** 我們的核心應用程式碼都集中在 `src/` 目錄下，職責清晰。

-   `src/train.py`:
    -   **職責**：模型訓練。
    -   **功能**：訓練一個 Scikit-learn 模型，並使用 `bentoml.sklearn.save_model()` 將其保存在 BentoML 的本地模型倉庫中。
-   `src/service.py`:
    -   **職責**：模型服務化。
    -   **功能**：載入訓練好的模型，並定義一個 API 端點。這是我們服務的核心邏輯。
-   `src/bentofile.yaml`:
    -   **職責**：服務打包設定檔。
    -   **功能**：它告訴 BentoML 這個服務的元數據、要打包哪些檔案 (`.py`)、需要哪些 Python 套件、以及依賴哪個模型。

---

## 第三章：手動執行 MLOps 工作流 (The Manual Workflow)

**原理**：**親手實踐是內化知識的最佳途徑。** 我們將使用 `Makefile` 這個專案的「儀表板」，手動模擬一次從無到有的完整流程。

請在 VS Code 的終端機中，從專案根目錄執行以下步驟。

### 步驟 3.1：安裝/更新依賴

**目標**：確保所有 Python 套件都已安裝。

```bash
make install
```

### 步驟 3.2：模型訓練

**目標**：產生一個可被服務的模型產物 (artifact)。

```bash
make train
```
-   **發生了什麼**：此指令執行了 `python src/train.py`。
-   **如何驗證**：執行 `bentoml models list`，你應該能看到一個名為 `iris_clf` 的模型。

### 步驟 3.3：啟動本地開發服務

**目標**：在本地快速啟動一個可供測試的 API 服務。

```bash
make run
```
-   **發生了什麼**：此指令執行了 `bentoml serve src/service.py:svc --reload`。`--reload` 參數非常方便，當你修改 `service.py` 時，服務會自動重啟。
-   **如何驗證**：在瀏覽器中打開 `http://127.0.0.1:3000`。你將看到 BentoML 自動生成的 API 文件頁面 (Swagger UI)。你可以在此頁面直接測試 API。

### 步驟 3.4：建置 Bento

**目標**：將您的服務、程式碼和模型依賴打包成一個標準化的、可部署的單元 (Bento)。

```bash
make bento-build
```
-   **發生了什麼**：此指令執行了 `bentoml build`。BentoML 會讀取 `src/bentofile.yaml`，找到所有必要的檔案和模型，並將它們打包在一起。
-   **如何驗證**：執行 `bentoml list`，你應該能看到一個名為 `iris_classifier` 的 Bento。

### 步驟 3.5：將 Bento 容器化

**目標**：將 Bento 轉換成一個標準的 Docker 映像檔，這是現代雲端部署的黃金標準。

```bash
make containerize
```
-   **發生了什麼**：此指令執行了 `bentoml containerize`。BentoML 會自動生成一個優化過的 `Dockerfile`，並為你建置 Docker 映像。
-   **如何驗證**：執行 `docker images | grep iris_classifier`，你應該能看到剛建置好的 Docker 映像。

**太棒了！你剛剛手動完成了一個完整的 MLOps 流程：從訓練 -> 本地服務 -> 打包 Bento -> 容器化。**

---

## 第四章：自動化的力量 (CI/CD)

**原理**：**CI/CD (持續整合/持續部署) 是 MLOps 的心臟。** 自動化流程確保每一次程式碼變更都經過同樣嚴格的步驟。

**功能**：`.github/workflows/cicd.yml` 定義了這個自動化流程。

### 這是如何運作的？

1.  **觸發 (Trigger)**：你使用 `git push` 將程式碼推送到 `main` 或 `GPU` 分支。
2.  **執行 (Execution)**：GitHub Actions 在雲端啟動一個虛擬機。
3.  **執行步驟 (Steps)**：此虛擬機將執行 `Makefile` 中的目標，模擬你剛剛手動操作的流程：
    -   `make install` (安裝依賴)
    -   `make lint` (檢查程式碼風格)
    -   `make format --check` (檢查程式碼格式)
    -   `make test` (執行測試)
    -   `make bento-build` (建置 Bento)
4.  **產物上傳 (Artifact Upload)**：
    -   如果以上所有步驟都成功，CI/CD 會將建置好的 Bento (`iris_classifier-....bento`) 打包並上傳。
    -   **這實現了可追溯性**：你可以隨時從 GitHub Actions 的紀錄中，下載任何一次程式碼提交所對應的、經過完整測試的可部署產物。

你不需要手動執行它，只需要知道，當你把程式碼推送到 GitHub，這一切都會在雲端自動、可靠地發生。

---

## 第五章：觸發自動化 CI/CD (Triggering Automation)

**原理**：**CI/CD (持續整合/持續部署) 的核心是事件驅動。** 我們設定當特定 Git 事件 (例如 `git push`) 發生時，GitHub Actions 會自動執行我們在 `.github/workflows/cicd.yml` 中定義好的所有流程。

現在，我們將把本地的專案與遠端的 GitHub 儲存庫 (Repository) 連接，並觸發第一次自動化流程。

### 步驟 5.1：在 GitHub 上建立新的儲存庫

1.  前往 [GitHub](https://github.com/new) 網站。
2.  建立一個新的**私有 (Private)** 或 **公開 (Public)** 儲存庫。**不要**勾選 "Initialize this repository with a README" 或其他檔案，因為我們的本地專案已經準備好了。
3.  建立後，複製儲存庫的 URL，格式通常是 `https://github.com/YourUsername/YourRepoName.git`。

### 步驟 5.2：將本地專案連接至遠端儲存庫

在 VS Code 的終端機中，依序執行以下指令：

1.  **初始化本地 Git 儲存庫 (若尚未執行)**
    *   這會將當前目錄變成一個 Git 可管理的專案。
    ```bash
    git init
    ```

2.  **將所有檔案加入 Git 追蹤**
    *   `.` 代表目前目錄下的所有檔案。
    ```bash
    git add .
    ```

3.  **建立第一次提交 (Commit)**
    *   `-m` 後面是這次提交的訊息，描述了我們做了什麼。
    ```bash
    git commit -m "feat: Initial MLOps project structure with BentoML"
    ```

4.  **將主要分支重新命名為 `main`**
    *   這是目前 Git 的標準做法。
    ```bash
    git branch -M main
    ```

5.  **連接遠端儲存庫**
    *   將 `<YOUR_REPO_URL>` 替換為你在上一步複製的 URL。
    ```bash
    git remote add origin <YOUR_REPO_URL>
    ```

6.  **推送 (Push) 你的程式碼**
    *   `-u` 參數會設定本地的 `main` 分支追蹤遠端的 `main` 分支。
    ```bash
    git push -u origin main
    ```

### 步驟 5.3：在 GitHub Actions 上觀察 CI/CD 流程

**恭喜！** 當你執行 `git push` 後，你就已經觸發了第一次的 CI/CD 流程。

1.  回到你的 GitHub 儲存庫頁面。
2.  點擊上方的 **"Actions"** 標籤頁。
3.  你應該會看到一個正在執行中的工作流程 (Workflow)，它的名稱是 "Python CI/CD"。
4.  點進去，你可以即時看到 GitHub 的伺服器正在按照 `cicd.yml` 的定義，一步一步地：
    *   安裝依賴
    *   執行 Linting
    *   執行程式碼格式檢查
    *   執行測試 (目前為空)
    *   **建立 Bento**
    *   **將建好的 Bento 存檔 (Archive Bento Build)**
5.  當流程成功跑完後 (顯示綠色勾勾 ✅)，點擊該次執行的摘要頁面，你可以在下方的 "Artifacts" 區域找到一個名為 `iris-classifier-bento` 的壓縮檔。你可以下載它，這就是你這次程式碼提交所產出的、可部署的模型服務！

你已經成功地建立並執行了一個完整的 MLOps 自動化流程。


## 第六章：下一步探索

你已經掌握了本專案的核心工作流程。接下來，你可以嘗試：

1.  **擴充服務**：在 `src/service.py` 中增加一個新的 API 端點，然後透過 `make run` 進行測試。
2.  **下載產物**：推送一次程式碼變更，等待 GitHub Actions 運行完畢，然後嘗試從運行紀錄中下載 Bento 產物。
3.  **探索部署**: `Makefile` 中的 `deploy` 目標是空的。你可以嘗試填充它，例如加入 `docker push` 指令，將 `make containerize` 建置的映像推送到 Docker Hub。
4.  **閱讀 CI/CD 設定檔**：打開 `.github/workflows/cicd.yml`，試著將裡面的指令與 `Makefile` 的目標對應起來。

**學習 MLOps 的旅程充滿挑戰與樂趣，而你已經成功建立了一個堅實的、自動化的基礎。祝賀你！**

---