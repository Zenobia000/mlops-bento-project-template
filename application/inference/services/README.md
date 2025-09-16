## BentoML Exploration

### List Local Models

`bentoml models list`

### Inspect a Saved Model

A saved BentoML model is typically composed of a `model.yaml` for metadata and a `saved_model.pkl` for the model artifact. You can inspect its contents:

`ls -l ~/.bentoml/models/iris_clf/<some_version_tag>/`

### Export a Model

Export a model from the local model store into a standalone `.bentomodel` file for sharing or archiving.

`bentoml models export iris_clf:latest .`