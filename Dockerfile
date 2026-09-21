# Inference image only. No notebooks inside.
FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TMPDIR=/var/tmp

# keep pip temp on a normal writable path inside the image
RUN mkdir -p /var/tmp && chmod 1777 /var/tmp

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY config/ config/
COPY src/ src/
COPY app/ app/
COPY data/sample_order.json data/sample_order.json

# model bytes and mlruns come from volumes at runtime
# so the container does not depend on a laptop path only
RUN mkdir -p models mlruns storage logs

EXPOSE 8000

CMD ["python", "-m", "app.run"]
