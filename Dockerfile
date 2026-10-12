FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    MNEMO_SQLITE_PATH=/data/mnemo.db

WORKDIR /app

# Dependencies first so code changes don't invalidate the (slow) install layer.
COPY pyproject.toml README.md LICENSE ./
RUN mkdir mnemo && touch mnemo/__init__.py \
    && pip install ".[server,postgres,mcp]" \
    && pip uninstall -y mnemo-memory

COPY mnemo ./mnemo
RUN pip install --no-deps .

# Bake the embedding model into the image: no download on cold start.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

RUN useradd --create-home mnemo && mkdir -p /data && chown -R mnemo /data /opt/fastembed
USER mnemo
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["mnemo", "serve", "--host", "0.0.0.0", "--port", "8000"]
