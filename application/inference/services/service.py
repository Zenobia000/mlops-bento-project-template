import numpy as np
import bentoml

# Load the model
iris_model_ref = bentoml.sklearn.get("iris_clf:latest")

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 20},
)
class IrisClassifier:

    @bentoml.api
    def classify(self, input_data: np.ndarray) -> np.ndarray:
        # Load the actual sklearn model from the BentoML model reference
        model = iris_model_ref.load_model()
        return model.predict(input_data)
