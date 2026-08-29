FROM python:3.11-slim

WORKDIR /app

# System dependencies required by OpenCV (headless build still needs these
# shared libs for image codec / GUI-stub support at import time)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app/ ./app/
COPY models/ ./models/

# database/ and uploads/ are created at runtime and expected to be mounted
# as volumes (see docker-compose.yml) so data persists across restarts.
RUN mkdir -p database uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]