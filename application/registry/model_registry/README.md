# Model Registry

This directory contains trained model artifacts and their metadata.

## Structure

```
model_registry/
├── README.md                 # This file
├── iris_classifier_YYYYMMDD_HHMMSS.joblib  # Trained model files
├── iris_classifier_YYYYMMDD_HHMMSS.json    # Model metrics and metadata
└── .gitkeep                  # Keep empty directory in git
```

## Model Storage Convention

- **Model files**: `{model_name}_{timestamp}.joblib`
- **Metadata files**: `{model_name}_{timestamp}.json`
- **Timestamp format**: `YYYYMMDD_HHMMSS`

## Metadata Format

Each `.json` file contains:
- Model performance metrics (accuracy, precision, recall, f1-score)
- Training parameters
- Dataset statistics
- Feature importance (when applicable)
- Confusion matrix
- Training timestamp

## Usage

Models are automatically saved here by the training pipeline:
```bash
poetry run python application/training/pipelines/iris_training_pipeline.py
```

Models can be loaded programmatically:
```python
import joblib
import json
from pathlib import Path

# Load model
model = joblib.load('path/to/model.joblib')

# Load metadata
with open('path/to/model.json', 'r') as f:
    metadata = json.load(f)
```

## Integration with BentoML

Models from this registry are used by BentoML services in `application/inference/services/`.
The latest model can be programmatically selected based on performance metrics.