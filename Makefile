start:
	uvicorn main:app --reload

dev_reqs:
	pip install -r requirements/dev.txt
