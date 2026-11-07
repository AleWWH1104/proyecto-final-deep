# Imagen del servicio: solo FastAPI + PyTorch CPU (sin pandas ni sklearn del entrenamiento).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=7860

WORKDIR /app

# Dependencias primero: esta capa se reutiliza mientras no cambie uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY app/ app/
COPY artefactos/ artefactos/
RUN uv sync --frozen --no-default-groups

# Hugging Face Spaces corre el contenedor con el usuario 1000; la base SQLite necesita una carpeta con permiso de escritura.
RUN useradd --uid 1000 --create-home usuario && mkdir -p datos && chown -R usuario /app/datos
USER usuario

EXPOSE 7860
# exec: fastapi queda como proceso principal y recibe las señales de apagado.
CMD ["sh", "-c", "exec fastapi run app/main.py --host 0.0.0.0 --port ${PORT}"]
