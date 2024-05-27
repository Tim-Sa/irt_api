start:
	venv/bin/python -m uvicorn app.main:app --reload

reqs:
	pip install -r requirements.txt

build:
	docker build -t irt_api .

run:
	docker run -d --name irt_app_container -p 8888:80 irt_api