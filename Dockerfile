FROM python:3.12-slim

WORKDIR /app

# Installation de uv
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY api/ api/
COPY monitoring/ monitoring/
COPY models/ models/

EXPOSE 7788

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]