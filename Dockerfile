FROM python:3.12-slim AS app-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app


RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY app ./app
COPY data ./data

# ---------------------------------------------------------------------------
# Entraînement (service one-shot)
# ---------------------------------------------------------------------------
FROM app-base AS trainer
CMD ["python", "-m", "src.train"]

# ---------------------------------------------------------------------------
# Application Streamlit
# ---------------------------------------------------------------------------
FROM app-base AS streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

# ---------------------------------------------------------------------------
# Airflow (api-server / scheduler / dag-processor)
# ---------------------------------------------------------------------------
FROM app-base AS airflow

ENV AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__DAGS_FOLDER=/app/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=false \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    PYTHONPATH=/app

RUN uv sync --frozen --no-dev --no-install-project --group airflow

COPY dags ./dags

# SimpleAuthManager (dev uniquement) : identifiants admin / admin
RUN mkdir -p /opt/airflow \
    && printf '{"admin": "admin"}' > /opt/airflow/simple_auth_manager_passwords.json.generated

EXPOSE 8080
CMD ["airflow", "api-server"]
