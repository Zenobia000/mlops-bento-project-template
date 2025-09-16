import numpy as np
import bentoml

# Load the model
iris_model_ref = bentoml.sklearn.get("iris_clf:latest")

@bentoml.Service(
    resources={"cpu": "2"},
    traffic={"timeout": 20},
)
class IrisClassifier:

    model = bentoml.models.get("iris_clf:latest")

    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model(self.model)

    @bentoml.api
    def classify(self, input_data: np.ndarray) -> np.ndarray:
        return self.model.predict(input_data)
