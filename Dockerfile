FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY api/ api/
COPY monitoring/ monitoring/

# Crée le dossier models/ vide si pas présent (le modèle sera monté via volume en prod)
RUN mkdir -p models/

EXPOSE 7788

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7788"]