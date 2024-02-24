start:
	uvicorn src.main:app --reload

dev_reqs:
	pip install -r requirements/dev.txt

test:
	make irt_test

irt_test:
	python src/irt/test.py
