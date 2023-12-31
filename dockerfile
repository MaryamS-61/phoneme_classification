FROM python:3.8.12-buster

WORKDIR /prod

COPY phoneme phoneme
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn phoneme.api.fast:app --host 0.0.0.0 --port $PORT
