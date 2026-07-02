FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir boto3==1.38.0

COPY handler.py run.py ./

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
