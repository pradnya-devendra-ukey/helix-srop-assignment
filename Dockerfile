FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Ingest docs at runtime (needs GOOGLE_API_KEY), then start the server
CMD ["sh", "-c", "python -m app.rag.ingest --path docs/ && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
