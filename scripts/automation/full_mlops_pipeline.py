#!/usr/bin/env python3
"""
完整 MLOps 流水線自動化腳本
執行從訓練到部署的完整流程
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import argparse

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class MLOpsPipeline:
    """MLOps 全流程自動化"""

    def __init__(self, config_path=None, dry_run=False):
        self.project_root = project_root
        self.dry_run = dry_run
        self.config = self.load_config(config_path)
        self.pipeline_results = {
            "pipeline_start": datetime.now().isoformat(),
            "steps": {}
        }

    def load_config(self, config_path):
        """載入流水線配置"""
        default_config = {
            "training": {
                "config_file": "application/training/configs/iris_config.json",
                "accuracy_threshold": 0.9,
                "performance_threshold_ms": 100
            },
            "validation": {
                "run_validation": True,
                "accuracy_threshold": 0.95
            },
            "service": {
                "build_service": True,
                "run_tests": True,
                "port": 3000
            },
            "deployment": {
                "environment": "staging",
                "auto_deploy": False
            },
            "monitoring": {
                "enable_metrics": True,
                "metrics_port": 8001
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # 合併配置
            for section in default_config:
                if section in user_config:
                    default_config[section].update(user_config[section])

        return default_config

    def run_command(self, command, cwd=None, description=""):
        """執行命令並記錄結果"""
        if cwd is None:
            cwd = self.project_root

        print(f"🔧 {description}")
        print(f"執行: {command}")

        if self.dry_run:
            print("(乾跑模式 - 未實際執行)")
            return True, "", ""

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=600  # 10分鐘超時
            )

            if result.returncode == 0:
                print("✅ 成功")
                return True, result.stdout, result.stderr
            else:
                print(f"❌ 失敗 (退出碼: {result.returncode})")
                print(f"錯誤輸出: {result.stderr}")
                return False, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            print("❌ 命令執行超時")
            return False, "", "命令執行超時"
        except Exception as e:
            print(f"❌ 執行錯誤: {str(e)}")
            return False, "", str(e)

    def step_1_model_training(self):
        """步驟 1: 模型訓練"""
        print("\n" + "="*60)
        print("📊 步驟 1: 模型訓練")
        print("="*60)

        config_file = self.config["training"]["config_file"]
        command = f"poetry run python application/training/pipelines/iris_training_pipeline.py --config {config_file}"

        success, stdout, stderr = self.run_command(
            command,
            description="訓練 Iris 分類模型"
        )

        self.pipeline_results["steps"]["training"] = {
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "stdout": stdout,
            "stderr": stderr
        }

        return success

    def step_2_model_validation(self):
        """步驟 2: 模型驗證"""
        print("\n" + "="*60)
        print("🎯 步驟 2: 模型驗證")
        print("="*60)

        if not self.config["validation"]["run_validation"]:
            print("⏭️ 跳過模型驗證")
            return True

        threshold = self.config["validation"]["accuracy_threshold"]
        command = f"poetry run python application/validation/model_validation/validate_model.py --model-path application/registry/model_registry/ --threshold {threshold}"

        success, stdout, stderr = self.run_command(
            command,
            description=f"驗證模型 (閾值: {threshold})"
        )

        self.pipeline_results["steps"]["validation"] = {
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "threshold": threshold,
            "stdout": stdout,
            "stderr": stderr
        }

        return success

    def step_3_build_service(self):
        """步驟 3: 建立 BentoML 服務"""
        print("\n" + "="*60)
        print("🚀 步驟 3: 建立 BentoML 服務")
        print("="*60)

        if not self.config["service"]["build_service"]:
            print("⏭️ 跳過服務建立")
            return True

        # 先確保服務文件存在並更新
        service_dir = self.project_root / "application" / "inference" / "services"

        # 建立 BentoML 服務
        command = "poetry run bentoml build"

        success, stdout, stderr = self.run_command(
            command,
            cwd=service_dir,
            description="建立 BentoML 服務"
        )

        self.pipeline_results["steps"]["build_service"] = {
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "stdout": stdout,
            "stderr": stderr
        }

        return success

    def step_4_test_service(self):
        """步驟 4: 測試服務"""
        print("\n" + "="*60)
        print("🧪 步驟 4: 服務測試")
        print("="*60)

        if not self.config["service"]["run_tests"]:
            print("⏭️ 跳過服務測試")
            return True

        # 啟動服務進行測試
        service_dir = self.project_root / "application" / "inference" / "services"
        port = self.config["service"]["port"]

        # 在背景啟動服務
        print("🔄 啟動測試服務...")
        if not self.dry_run:
            service_process = subprocess.Popen(
                ["poetry", "run", "bentoml", "serve", "iris_service:IrisClassifier", "--port", str(port)],
                cwd=service_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # 等待服務啟動
            time.sleep(10)

            try:
                # 運行基本測試
                test_command = f"poetry run python application/inference/services/test_service.py"
                test_success, test_stdout, test_stderr = self.run_command(
                    test_command,
                    description="執行服務功能測試"
                )

                if test_success:
                    # 運行負載測試
                    load_test_command = f"poetry run python tests/integration/test_load_performance.py --url http://localhost:{port} --concurrent 5 --requests 10"
                    load_success, load_stdout, load_stderr = self.run_command(
                        load_test_command,
                        description="執行服務負載測試"
                    )
                else:
                    load_success = False
                    load_stdout = ""
                    load_stderr = "功能測試失敗，跳過負載測試"

                overall_success = test_success and load_success

            finally:
                # 關閉服務
                print("🔄 關閉測試服務...")
                service_process.terminate()
                service_process.wait(timeout=10)

        else:
            print("(乾跑模式 - 跳過實際服務測試)")
            overall_success = True
            test_stdout = load_stdout = "乾跑模式"
            test_stderr = load_stderr = ""

        self.pipeline_results["steps"]["test_service"] = {
            "success": overall_success,
            "timestamp": datetime.now().isoformat(),
            "functional_test": {
                "success": test_success if not self.dry_run else True,
                "stdout": test_stdout,
                "stderr": test_stderr
            },
            "load_test": {
                "success": load_success if not self.dry_run else True,
                "stdout": load_stdout,
                "stderr": load_stderr
            }
        }

        return overall_success

    def step_5_deployment_preparation(self):
        """步驟 5: 部署準備"""
        print("\n" + "="*60)
        print("📦 步驟 5: 部署準備")
        print("="*60)

        environment = self.config["deployment"]["environment"]

        # 匯出 BentoML 服務
        service_dir = self.project_root / "application" / "inference" / "services"
        export_command = f"poetry run bentoml export iris_classifier:latest ./iris_service_{environment}.bento"

        success, stdout, stderr = self.run_command(
            export_command,
            cwd=service_dir,
            description=f"匯出服務至 {environment} 環境"
        )

        # 創建部署配置
        if success and not self.dry_run:
            deployment_config = {
                "environment": environment,
                "service_name": "iris_classifier",
                "export_time": datetime.now().isoformat(),
                "config": self.config["deployment"]
            }

            config_path = service_dir / f"deployment_config_{environment}.json"
            with open(config_path, 'w') as f:
                json.dump(deployment_config, f, indent=2)

        self.pipeline_results["steps"]["deployment_prep"] = {
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "environment": environment,
            "stdout": stdout,
            "stderr": stderr
        }

        return success

    def step_6_deploy(self):
        """步驟 6: 自動部署 (可選)"""
        print("\n" + "="*60)
        print("🚀 步驟 6: 自動部署")
        print("="*60)

        if not self.config["deployment"]["auto_deploy"]:
            print("⏭️ 跳過自動部署 (需手動部署)")
            return True

        environment = self.config["deployment"]["environment"]

        # 這裡可以添加實際的部署邏輯
        # 例如：部署到 AWS ECS, Google Cloud Run, 等

        print(f"🔄 部署到 {environment} 環境...")
        print("💡 提示: 在此處添加您的部署邏輯")

        # 模擬部署
        if not self.dry_run:
            time.sleep(2)

        self.pipeline_results["steps"]["deployment"] = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "environment": environment,
            "note": "部署邏輯需要根據實際環境配置"
        }

        return True

    def generate_report(self):
        """生成流水線報告"""
        self.pipeline_results["pipeline_end"] = datetime.now().isoformat()

        # 計算總體成功狀態
        all_steps_success = all(
            step_result.get("success", False)
            for step_result in self.pipeline_results["steps"].values()
        )

        self.pipeline_results["overall_success"] = all_steps_success

        # 保存報告
        report_filename = f"mlops_pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.project_root / "reports" / report_filename
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(self.pipeline_results, f, indent=2, default=str)

        print("\n" + "="*60)
        print("📋 流水線執行總結")
        print("="*60)
        print(f"整體狀態: {'✅ 成功' if all_steps_success else '❌ 失敗'}")
        print(f"報告位置: {report_path}")

        # 顯示各步驟狀態
        for step_name, step_result in self.pipeline_results["steps"].items():
            status = "✅" if step_result.get("success") else "❌"
            print(f"{status} {step_name}")

        return all_steps_success, report_path

    def run_full_pipeline(self):
        """執行完整流水線"""
        print("🚀 開始 MLOps 完整流水線")
        print(f"模式: {'乾跑' if self.dry_run else '實際執行'}")
        print("="*60)

        steps = [
            ("模型訓練", self.step_1_model_training),
            ("模型驗證", self.step_2_model_validation),
            ("建立服務", self.step_3_build_service),
            ("測試服務", self.step_4_test_service),
            ("部署準備", self.step_5_deployment_preparation),
            ("自動部署", self.step_6_deploy)
        ]

        for step_name, step_function in steps:
            try:
                success = step_function()
                if not success:
                    print(f"\n❌ 流水線在 '{step_name}' 步驟失敗")
                    if not self.dry_run:
                        break
            except Exception as e:
                print(f"\n❌ 步驟 '{step_name}' 執行時發生異常: {str(e)}")
                self.pipeline_results["steps"][step_name.lower().replace(" ", "_")] = {
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
                if not self.dry_run:
                    break

        # 生成報告
        overall_success, report_path = self.generate_report()

        if overall_success:
            print(f"\n🎉 MLOps 流水線執行完成！")
        else:
            print(f"\n⚠️ MLOps 流水線執行遇到問題，請查看報告")

        return overall_success

def main():
    parser = argparse.ArgumentParser(description="MLOps 完整流水線自動化")
    parser.add_argument("--config", type=str, help="流水線配置文件路徑")
    parser.add_argument("--dry-run", action="store_true", help="乾跑模式 (不實際執行)")
    args = parser.parse_args()

    # 執行流水線
    pipeline = MLOpsPipeline(config_path=args.config, dry_run=args.dry_run)
    success = pipeline.run_full_pipeline()

    # 設置退出碼
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()