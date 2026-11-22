FROM python:3.12-slim
WORKDIR /opt/worker
RUN pip install --no-cache-dir "psycopg[binary]==3.2.3" "httpx==0.28.1"
COPY *.py ./
