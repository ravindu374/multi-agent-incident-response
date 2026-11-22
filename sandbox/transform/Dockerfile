FROM python:3.12-slim
ENV DBT_SEND_ANONYMOUS_USAGE_STATS=false
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dbt /dbt
COPY runner.py .
CMD ["python", "-u", "runner.py"]
