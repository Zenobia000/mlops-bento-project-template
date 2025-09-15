import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import joblib

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class IrisTrainingPipeline:
    """Iris 分類模型訓練流水線"""

    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.setup_mlflow()

    def load_config(self, config_path):
        """載入訓練配置"""
        default_config = {
            "model": {
                "n_estimators": 100,
                "max_depth": None,
                "random_state": 42
            },
            "data": {
                "test_size": 0.2,
                "random_state": 42
            },
            "experiment_name": "iris_classification_pipeline"
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            # 合併配置
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
            return config

        return default_config

    def setup_mlflow(self):
        """設置 MLflow"""
        mlflow.set_experiment(self.config["experiment_name"])

    def load_data(self):
        """載入和準備數據"""
        print("📊 載入數據...")
        iris = load_iris()
        X, y = iris.data, iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=y
        )

        print(f"訓練集大小: {X_train.shape}")
        print(f"測試集大小: {X_test.shape}")

        return X_train, X_test, y_train, y_test, iris.target_names

    def train_model(self, X_train, y_train):
        """訓練模型"""
        print("🔧 訓練模型...")
        model = RandomForestClassifier(**self.config["model"])
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test, target_names):
        """評估模型"""
        print("📈 評估模型...")
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }

    def save_model(self, model, metrics, model_name="iris_classifier"):
        """儲存模型和指標"""
        print("💾 儲存模型...")

        # 創建模型目錄
        model_dir = project_root / "application" / "registry" / "model_registry"
        model_dir.mkdir(parents=True, exist_ok=True)

        # 儲存模型
        model_path = model_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(model, model_path)

        # 儲存指標
        metrics_path = model_path.with_suffix('.json')
        with open(metrics_path, 'w') as f:
            # 將 numpy 數組轉換為列表
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                else:
                    serializable_metrics[key] = value
            json.dump(serializable_metrics, f, indent=2)

        return str(model_path), str(metrics_path)

    def run_training(self):
        """執行完整的訓練流水線"""
        print("🚀 開始 MLOps 訓練流水線...")

        with mlflow.start_run(run_name=f"iris_pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # 載入數據
            X_train, X_test, y_train, y_test, target_names = self.load_data()

            # 記錄參數
            mlflow.log_params(self.config["model"])
            mlflow.log_params(self.config["data"])

            # 訓練模型
            model = self.train_model(X_train, y_train)

            # 評估模型
            metrics = self.evaluate_model(model, X_test, y_test, target_names)

            # 記錄指標
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("precision_macro", metrics['classification_report']['macro avg']['precision'])
            mlflow.log_metric("recall_macro", metrics['classification_report']['macro avg']['recall'])
            mlflow.log_metric("f1_macro", metrics['classification_report']['macro avg']['f1-score'])

            # 記錄模型到 MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="iris_classifier_pipeline"
            )

            # 儲存模型到本地
            model_path, metrics_path = self.save_model(model, metrics)

            print("✅ 訓練完成！")
            print(f"📊 準確率: {metrics['accuracy']:.4f}")
            print(f"💾 模型儲存位置: {model_path}")
            print(f"📈 指標儲存位置: {metrics_path}")

            return model, metrics

def main():
    parser = argparse.ArgumentParser(description="Iris 分類模型訓練流水線")
    parser.add_argument("--config", type=str, help="訓練配置文件路徑")
    args = parser.parse_args()

    # 執行訓練流水線
    pipeline = IrisTrainingPipeline(config_path=args.config)
    model, metrics = pipeline.run_training()

    return model, metrics

if __name__ == "__main__":
    main()