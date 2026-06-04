FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# If a wheels/ directory exists (air-gapped), install from it; otherwise use PyPI
COPY wheels* /tmp/wheels/
RUN if [ -d /tmp/wheels ] && [ "$(ls -A /tmp/wheels 2>/dev/null)" ]; then \
        pip install --no-cache-dir --no-index --find-links=/tmp/wheels -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi && \
    rm -rf /tmp/wheels

COPY src/ src/

EXPOSE 8088

ENTRYPOINT ["python", "-m", "src.main"]
