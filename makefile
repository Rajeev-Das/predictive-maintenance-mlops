# Makefile for Predictive Maintenance MLOps

.PHONY: venv train evaluate api api-test lint format test clean

venv:
	python -m venv venv
	. venv/bin/activate; pip install -r requirements.txt

train:
	python main.py --mode train

evaluate:
	python main.py --mode evaluate

api:
	uvicorn api:app --reload

api-test:
	python api_client.py

lint:
	PYTHONPATH=. pylint src/ api.py api_client.py

format:
	black src/ api.py api_client.py

test:
	PYTHONPATH=. pytest

clean:
	rm -rf __pycache__ .pytest_cache model/*.keras model/*.joblib
