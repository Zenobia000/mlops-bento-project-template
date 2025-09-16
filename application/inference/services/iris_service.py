import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_model_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service(
    name="iris_classifier",
    runners=[iris_model_runner],
)

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> np.ndarray:
    return await iris_model_runner.predict.async_run(input_series)