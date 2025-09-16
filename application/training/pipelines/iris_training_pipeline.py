import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import joblib

# Add project root directory to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import mlflow
import mlflow.sklearn
import bentoml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class IrisTrainingPipeline:
    """Iris Classification Model Training Pipeline"""

    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.setup_mlflow()

    def load_config(self, config_path):
        """Load training configuration"""
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
            # Merge configurations
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
            return config

        return default_config

    def setup_mlflow(self):
        """Setup MLflow"""
        # Set the tracking server URI to connect to the MLflow container
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        # Set credentials for the MinIO artifact store
        os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9010"

        print("🔧 MLflow configured:")
        print(f"   - Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"   - S3 Endpoint URL: {os.environ.get('MLFLOW_S3_ENDPOINT_URL')}")

        mlflow.set_experiment(self.config["experiment_name"])

    def load_data(self):
        """Load and prepare data"""
        print("📊 Loading data...")
        iris = load_iris()
        X, y = iris.data, iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=y
        )

        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")

        return X_train, X_test, y_train, y_test, iris.target_names

    def train_model(self, X_train, y_train):
        """Train model"""
        print("🔧 Training model...")
        model = RandomForestClassifier(**self.config["model"])
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test, target_names):
        """Evaluate model"""
        print("📈 Evaluating model...")
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
        """Save model and metrics"""
        print("💾 Saving model...")

        # Create model directory
        model_dir = project_root / "application" / "registry" / "model_registry"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(model, model_path)

        # Save metrics
        metrics_path = model_path.with_suffix('.json')
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                else:
                    serializable_metrics[key] = value
            json.dump(serializable_metrics, f, indent=2)

        return str(model_path), str(metrics_path)

    def import_model_to_bentoml(self, model, metrics, model_name="iris_clf"):
        """Import the trained model into the BentoML model store."""
        print(f"🍱 Importing model '{model_name}' to BentoML store...")

        # Save the model to the BentoML store
        bento_model = bentoml.sklearn.save_model(
            model_name,
            model,
            signatures={
                "predict": {"batchable": True, "batch_dim": 0},
            },
            labels={
                "owner": "mlops-team",
                "stage": "dev",
                "accuracy": f"{metrics['accuracy']:.4f}"
            }
        )
        print(f"✅ Model imported to BentoML: {bento_model.tag}")
        return bento_model

    def run_training(self):
        """Execute complete training pipeline"""
        print("🚀 Starting MLOps training pipeline...")

        with mlflow.start_run(run_name=f"iris_pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Load data
            X_train, X_test, y_train, y_test, target_names = self.load_data()

            # Log parameters
            mlflow.log_params(self.config["model"])
            mlflow.log_params(self.config["data"])

            # Train model
            model = self.train_model(X_train, y_train)

            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test, target_names)

            # Log metrics
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("precision_macro", metrics['classification_report']['macro avg']['precision'])
            mlflow.log_metric("recall_macro", metrics['classification_report']['macro avg']['recall'])
            mlflow.log_metric("f1_macro", metrics['classification_report']['macro avg']['f1-score'])

            # Log model to MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="iris_classifier_pipeline"
            )

            # Save model locally
            model_path, metrics_path = self.save_model(model, metrics)

            # Import model to BentoML store
            self.import_model_to_bentoml(model, metrics)

            print("✅ Training completed!")
            print(f"📊 Accuracy: {metrics['accuracy']:.4f}")
            print(f"💾 Model saved to: {model_path}")
            print(f"📈 Metrics saved to: {metrics_path}")

            return model, metrics

def main():
    parser = argparse.ArgumentParser(description="Iris Classification Model Training Pipeline")
    parser.add_argument("--config", type=str, help="Training configuration file path")
    args = parser.parse_args()

    # Execute training pipeline
    pipeline = IrisTrainingPipeline(config_path=args.config)
    model, metrics = pipeline.run_training()

    return model, metrics

if __name__ == "__main__":
    main()