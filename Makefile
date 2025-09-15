.PHONY: install train test format lint bento-build containerize run deploy all

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
	#force install latest whisper if needed, commented out for now
	# pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

train:
	python src/train.py

test:
	# No application-specific tests yet. Add them here.
	# python -m pytest -vv --cov=src test/test_service.py
	echo "No application tests found."

format:	
	black src/*.py

lint:
	pylint --disable=R,C src/*.py

bento-build:
	bentoml build -f src/bentofile.yaml src

containerize:
	bentoml containerize iris_classifier:latest

run:
	bentoml serve src/service.py:svc --reload

deploy:
	# Deployment steps go here, e.g., pushing the container to a registry.
	echo "Deployment target is a placeholder."
		
all: install lint test format train bento-build
