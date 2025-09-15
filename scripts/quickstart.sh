#!/bin/bash

# MLOps 快速入門腳本
# 一鍵設置和運行完整的 MLOps 流水線

set -e  # 遇到錯誤時退出

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印帶顏色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo
    echo "=================================================="
    print_message $CYAN "$1"
    echo "=================================================="
}

print_step() {
    print_message $BLUE "🔧 $1"
}

print_success() {
    print_message $GREEN "✅ $1"
}

print_warning() {
    print_message $YELLOW "⚠️  $1"
}

print_error() {
    print_message $RED "❌ $1"
}

# 檢查必要工具
check_requirements() {
    print_header "檢查系統需求"

    # 檢查 Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python: $PYTHON_VERSION"
    else
        print_error "Python 3 未安裝"
        exit 1
    fi

    # 檢查 Poetry
    if command -v poetry &> /dev/null; then
        POETRY_VERSION=$(poetry --version | cut -d' ' -f3)
        print_success "Poetry: $POETRY_VERSION"
    else
        print_warning "Poetry 未安裝，將嘗試安裝..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # 檢查 Git
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        print_success "Git: $GIT_VERSION"
    else
        print_error "Git 未安裝"
        exit 1
    fi

    # 檢查 Docker (可選)
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
        print_success "Docker: $DOCKER_VERSION"
        DOCKER_AVAILABLE=true
    else
        print_warning "Docker 未安裝 (容器化功能將不可用)"
        DOCKER_AVAILABLE=false
    fi

    # 檢查 NVIDIA GPU (可選)
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU 可用"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -1
        GPU_AVAILABLE=true
    else
        print_warning "NVIDIA GPU 不可用 (將使用 CPU)"
        GPU_AVAILABLE=false
    fi
}

# 安裝依賴
install_dependencies() {
    print_header "安裝專案依賴"

    print_step "配置 Poetry..."
    poetry config virtualenvs.create true
    poetry config virtualenvs.in-project false

    print_step "安裝 Python 依賴..."
    if poetry install --with dev --extras all; then
        print_success "依賴安裝完成"
    else
        print_error "依賴安裝失敗"
        exit 1
    fi

    print_step "安裝 GPU 相關依賴..."
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        poetry run pip install tensorflow[and-cuda]
    else
        poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        poetry run pip install tensorflow-cpu
    fi

    # 安裝 OpenAI Whisper
    poetry run pip install git+https://github.com/openai/whisper.git

    print_success "所有依賴安裝完成"
}

# 驗證環境
verify_environment() {
    print_header "驗證開發環境"

    print_step "測試 GPU 可用性..."
    if poetry run python shared/utils/gpu_utils/verify_cuda_pytorch.py 2>/dev/null; then
        print_success "PyTorch GPU 支援正常"
    else
        print_warning "PyTorch GPU 支援不可用，使用 CPU"
    fi

    print_step "測試依賴導入..."
    poetry run python -c "
import numpy as np
import pandas as pd
import sklearn
import bentoml
import mlflow
print('所有核心依賴導入成功')
" && print_success "依賴測試通過"
}

# 運行完整 MLOps 流水線
run_mlops_pipeline() {
    print_header "執行 MLOps 流水線"

    # 確保目錄存在
    mkdir -p application/registry/model_registry
    mkdir -p reports

    print_step "步驟 1: 訓練模型..."
    if poetry run python application/training/pipelines/iris_training_pipeline.py --config application/training/configs/iris_config.json; then
        print_success "模型訓練完成"
    else
        print_error "模型訓練失敗"
        return 1
    fi

    print_step "步驟 2: 驗證模型..."
    if poetry run python application/validation/model_validation/validate_model.py --model-path application/registry/model_registry/ --threshold 0.9; then
        print_success "模型驗證通過"
    else
        print_warning "模型驗證失敗，但繼續流程"
    fi

    print_step "步驟 3: 建立 BentoML 服務..."
    cd application/inference/services
    if poetry run bentoml build; then
        print_success "BentoML 服務建立完成"
    else
        print_error "BentoML 服務建立失敗"
        cd ../../../
        return 1
    fi
    cd ../../../

    print_success "MLOps 流水線執行完成！"
}

# 啟動服務
start_services() {
    print_header "啟動服務"

    print_step "啟動 BentoML 推論服務..."
    cd application/inference/services

    # 在背景啟動服務
    nohup poetry run bentoml serve iris_service:IrisClassifier --port 3000 > ../../../bentoml_service.log 2>&1 &
    BENTOML_PID=$!
    echo $BENTOML_PID > ../../../bentoml.pid

    cd ../../../

    print_step "等待服務啟動..."
    sleep 10

    # 檢查服務是否啟動成功
    if curl -s http://localhost:3000/health_check > /dev/null; then
        print_success "BentoML 服務已啟動 (PID: $BENTOML_PID)"
        print_message $CYAN "服務地址: http://localhost:3000"
        print_message $CYAN "健康檢查: http://localhost:3000/health_check"
    else
        print_error "服務啟動失敗，查看日誌: tail -f bentoml_service.log"
        return 1
    fi
}

# 運行測試
run_tests() {
    print_header "執行服務測試"

    print_step "功能測試..."
    if poetry run python application/inference/services/test_service.py; then
        print_success "功能測試通過"
    else
        print_warning "功能測試失敗"
    fi

    print_step "負載測試..."
    if poetry run python tests/integration/test_load_performance.py --concurrent 3 --requests 5; then
        print_success "負載測試完成"
    else
        print_warning "負載測試遇到問題"
    fi
}

# 清理函數
cleanup() {
    print_header "清理資源"

    # 停止 BentoML 服務
    if [[ -f bentoml.pid ]]; then
        BENTOML_PID=$(cat bentoml.pid)
        if kill $BENTOML_PID 2>/dev/null; then
            print_success "BentoML 服務已停止"
        fi
        rm -f bentoml.pid
    fi

    # 清理日誌文件
    if [[ -f bentoml_service.log ]]; then
        mv bentoml_service.log logs/bentoml_service_$(date +%Y%m%d_%H%M%S).log 2>/dev/null || rm -f bentoml_service.log
    fi
}

# 顯示使用說明
show_usage() {
    print_header "使用說明"
    echo "用法: $0 [選項]"
    echo
    echo "選項:"
    echo "  --install-only     只安裝依賴，不運行流水線"
    echo "  --train-only       只運行訓練，不啟動服務"
    echo "  --no-tests         跳過測試步驟"
    echo "  --no-docker        跳過 Docker 相關步驟"
    echo "  --help             顯示此說明"
    echo
    echo "範例:"
    echo "  $0                 # 運行完整流程"
    echo "  $0 --install-only  # 只安裝依賴"
    echo "  $0 --train-only    # 只訓練模型"
}

# 主函數
main() {
    # 解析命令行參數
    INSTALL_ONLY=false
    TRAIN_ONLY=false
    NO_TESTS=false
    NO_DOCKER=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-only)
                INSTALL_ONLY=true
                shift
                ;;
            --train-only)
                TRAIN_ONLY=true
                shift
                ;;
            --no-tests)
                NO_TESTS=true
                shift
                ;;
            --no-docker)
                NO_DOCKER=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "未知選項: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # 設置 trap 來處理清理
    trap cleanup EXIT

    print_message $PURPLE "🚀 MLOps 完整流水線快速入門"
    print_message $PURPLE "================================"

    # 檢查需求
    check_requirements

    # 安裝依賴
    install_dependencies

    if [[ "$INSTALL_ONLY" == "true" ]]; then
        print_success "依賴安裝完成，退出"
        exit 0
    fi

    # 驗證環境
    verify_environment

    # 運行 MLOps 流水線
    if ! run_mlops_pipeline; then
        print_error "MLOps 流水線執行失敗"
        exit 1
    fi

    if [[ "$TRAIN_ONLY" == "true" ]]; then
        print_success "訓練完成，退出"
        exit 0
    fi

    # 啟動服務
    if ! start_services; then
        print_error "服務啟動失敗"
        exit 1
    fi

    # 運行測試
    if [[ "$NO_TESTS" != "true" ]]; then
        run_tests
    fi

    print_header "🎉 完成！"
    print_success "MLOps 流水線已成功運行"
    echo
    print_message $CYAN "服務資訊:"
    print_message $CYAN "  - BentoML 服務: http://localhost:3000"
    print_message $CYAN "  - API 文檔: http://localhost:3000/docs"
    print_message $CYAN "  - 健康檢查: http://localhost:3000/health_check"
    echo
    print_message $YELLOW "下一步:"
    print_message $YELLOW "  - 查看服務日誌: tail -f bentoml_service.log"
    print_message $YELLOW "  - 測試 API: curl http://localhost:3000/health_check"
    print_message $YELLOW "  - 停止服務: kill \$(cat bentoml.pid)"
    echo
    print_message $CYAN "查看完整教學: MLOPS_COMPLETE_TUTORIAL.md"
}

# 執行主函數
main "$@"