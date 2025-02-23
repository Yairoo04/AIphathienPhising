FROM python:3.9-slim

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app

CMD ["python", "app/main.py"]
