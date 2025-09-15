#!/usr/bin/env python3
"""
模型驗證腳本
用於驗證新訓練的模型是否滿足部署標準
"""

import sys
import argparse
import json
import joblib
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class ModelValidator:
    """模型驗證器"""

    def __init__(self, accuracy_threshold=0.9, performance_threshold_ms=100):
        self.accuracy_threshold = accuracy_threshold
        self.performance_threshold_ms = performance_threshold_ms
        self.validation_results = {}

    def load_latest_model(self, model_registry_path):
        """載入最新的模型"""
        registry_path = Path(model_registry_path)

        if not registry_path.exists():
            raise FileNotFoundError(f"模型註冊表目錄不存在: {registry_path}")

        # 尋找最新的模型文件
        model_files = list(registry_path.glob("*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"在 {registry_path} 中未找到模型文件")

        # 按修改時間排序，獲取最新的
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)

        print(f"📦 載入模型: {latest_model_file}")
        model = joblib.load(latest_model_file)

        # 嘗試載入對應的指標文件
        metrics_file = latest_model_file.with_suffix('.json')
        metrics = None
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

        return model, metrics, latest_model_file

    def validate_accuracy(self, model):
        """驗證模型準確率"""
        print("🎯 驗證模型準確率...")

        # 載入測試資料
        iris = load_iris()
        X, y = iris.data, iris.target

        # 使用不同的隨機種子創建驗證集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=123, stratify=y
        )

        # 預測
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        # 生成分類報告
        report = classification_report(
            y_val, y_pred,
            target_names=iris.target_names,
            output_dict=True
        )

        validation_result = {
            "accuracy": accuracy,
            "threshold": self.accuracy_threshold,
            "passed": accuracy >= self.accuracy_threshold,
            "classification_report": report
        }

        self.validation_results["accuracy_validation"] = validation_result

        print(f"準確率: {accuracy:.4f} (閾值: {self.accuracy_threshold})")
        print(f"結果: {'✅ 通過' if validation_result['passed'] else '❌ 未通過'}")

        return validation_result

    def validate_performance(self, model, num_samples=1000):
        """驗證模型推論性能"""
        print("⚡ 驗證模型推論性能...")

        # 生成測試資料
        X_test = np.random.rand(num_samples, 4)

        # 測量推論時間
        import time

        start_time = time.time()
        predictions = model.predict(X_test)
        end_time = time.time()

        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_sample_ms = total_time_ms / num_samples

        validation_result = {
            "total_time_ms": total_time_ms,
            "avg_time_per_sample_ms": avg_time_per_sample_ms,
            "samples_tested": num_samples,
            "threshold_ms": self.performance_threshold_ms,
            "passed": avg_time_per_sample_ms <= self.performance_threshold_ms
        }

        self.validation_results["performance_validation"] = validation_result

        print(f"平均推論時間: {avg_time_per_sample_ms:.4f} ms/樣本 (閾值: {self.performance_threshold_ms} ms)")
        print(f"結果: {'✅ 通過' if validation_result['passed'] else '❌ 未通過'}")

        return validation_result

    def validate_model_structure(self, model):
        """驗證模型結構"""
        print("🏗️ 驗證模型結構...")

        validation_result = {
            "model_type": type(model).__name__,
            "has_predict_method": hasattr(model, 'predict'),
            "has_predict_proba_method": hasattr(model, 'predict_proba'),
            "feature_importances_available": hasattr(model, 'feature_importances_')
        }

        # 檢查必要的方法
        required_methods = ['predict', 'predict_proba']
        missing_methods = [method for method in required_methods if not hasattr(model, method)]

        validation_result["missing_methods"] = missing_methods
        validation_result["passed"] = len(missing_methods) == 0

        self.validation_results["structure_validation"] = validation_result

        print(f"模型類型: {validation_result['model_type']}")
        print(f"必要方法檢查: {'✅ 通過' if validation_result['passed'] else '❌ 未通過'}")
        if missing_methods:
            print(f"缺少方法: {missing_methods}")

        return validation_result

    def validate_input_output(self, model):
        """驗證輸入輸出格式"""
        print("🔄 驗證輸入輸出格式...")

        validation_result = {"passed": True, "errors": []}

        try:
            # 測試正常輸入
            test_input = np.array([[5.1, 3.5, 1.4, 0.2]])
            prediction = model.predict(test_input)
            probabilities = model.predict_proba(test_input)

            # 驗證輸出格式
            if not isinstance(prediction, np.ndarray):
                validation_result["errors"].append("predict() 應返回 numpy array")
                validation_result["passed"] = False

            if len(prediction) != 1:
                validation_result["errors"].append("預測結果數量應與輸入樣本數相等")
                validation_result["passed"] = False

            if not isinstance(probabilities, np.ndarray):
                validation_result["errors"].append("predict_proba() 應返回 numpy array")
                validation_result["passed"] = False

            if probabilities.shape != (1, 3):  # 1 樣本, 3 類別
                validation_result["errors"].append("概率輸出形狀不正確")
                validation_result["passed"] = False

            # 測試邊界情況
            extreme_input = np.array([[0.0, 0.0, 0.0, 0.0]])
            model.predict(extreme_input)
            model.predict_proba(extreme_input)

            validation_result.update({
                "prediction_shape": prediction.shape,
                "probabilities_shape": probabilities.shape,
                "prediction_dtype": str(prediction.dtype),
                "probabilities_dtype": str(probabilities.dtype)
            })

        except Exception as e:
            validation_result["errors"].append(f"輸入輸出測試失敗: {str(e)}")
            validation_result["passed"] = False

        self.validation_results["input_output_validation"] = validation_result

        print(f"結果: {'✅ 通過' if validation_result['passed'] else '❌ 未通過'}")
        if validation_result["errors"]:
            for error in validation_result["errors"]:
                print(f"  錯誤: {error}")

        return validation_result

    def run_full_validation(self, model_registry_path):
        """運行完整的模型驗證"""
        print("🚀 開始模型驗證流程...")
        print("=" * 60)

        try:
            # 載入模型
            model, metrics, model_file = self.load_latest_model(model_registry_path)

            # 運行各項驗證
            accuracy_result = self.validate_accuracy(model)
            performance_result = self.validate_performance(model)
            structure_result = self.validate_model_structure(model)
            io_result = self.validate_input_output(model)

            # 計算整體結果
            all_validations = [
                accuracy_result["passed"],
                performance_result["passed"],
                structure_result["passed"],
                io_result["passed"]
            ]

            overall_passed = all(all_validations)

            # 準備最終報告
            validation_summary = {
                "model_file": str(model_file),
                "validation_timestamp": datetime.now().isoformat(),
                "overall_passed": overall_passed,
                "validations": self.validation_results,
                "summary": {
                    "accuracy_passed": accuracy_result["passed"],
                    "performance_passed": performance_result["passed"],
                    "structure_passed": structure_result["passed"],
                    "input_output_passed": io_result["passed"]
                }
            }

            # 保存驗證報告
            report_path = model_file.parent / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(validation_summary, f, indent=2, default=str)

            print("=" * 60)
            print("📋 驗證總結:")
            print(f"整體結果: {'✅ 通過所有驗證' if overall_passed else '❌ 存在驗證失敗'}")
            print(f"驗證報告: {report_path}")

            return overall_passed, validation_summary

        except Exception as e:
            print(f"❌ 驗證過程中發生錯誤: {str(e)}")
            return False, {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="模型驗證工具")
    parser.add_argument("--model-path", type=str, required=True,
                      help="模型註冊表目錄路徑")
    parser.add_argument("--threshold", type=float, default=0.9,
                      help="準確率閾值 (預設: 0.9)")
    parser.add_argument("--performance-threshold", type=float, default=100,
                      help="性能閾值 ms (預設: 100)")

    args = parser.parse_args()

    # 創建驗證器並運行驗證
    validator = ModelValidator(
        accuracy_threshold=args.threshold,
        performance_threshold_ms=args.performance_threshold
    )

    passed, results = validator.run_full_validation(args.model_path)

    # 根據驗證結果設置退出碼
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()