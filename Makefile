.PHONY: install data train test api

install:
	pip install -r requirements.txt

data:
	cd data && python generate_data.py

train:
	python train.py

test:
	pytest src/tests/ -v --tb=short

api:
	uvicorn src.api.app:app --reload --port 8000

all: install data train test
