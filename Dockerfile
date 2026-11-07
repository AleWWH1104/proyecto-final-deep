# Imagen del servicio: solo FastAPI + PyTorch CPU (sin pandas ni sklearn del entrenamiento).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH" \
    # Un hilo para PyTorch: con la fracción de CPU del plan gratuito, varios hilos se frenan entre sí (picos de segundos).
    OMP_NUM_THREADS=1 \
    PORT=8000

WORKDIR /app

# Dependencias primero: esta capa se reutiliza mientras no cambie uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY app/ app/
COPY artefactos/ artefactos/
RUN uv sync --frozen --no-default-groups

# Sin root dentro del contenedor; la base SQLite necesita una carpeta con permiso de escritura.
RUN useradd --uid 1000 --create-home usuario && mkdir -p datos && chown -R usuario /app/datos
USER usuario

# Render define PORT al arrancar; en local se usa 8000.
EXPOSE 8000
# exec: fastapi queda como proceso principal y recibe las señales de apagado.
CMD ["sh", "-c", "exec fastapi run app/main.py --host 0.0.0.0 --port ${PORT}"]
