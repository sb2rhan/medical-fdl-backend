FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Artifacts should be mounted or baked in (see Step 10)
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]

**`requirements.txt`**
fastapi>=0.111
uvicorn[standard]>=0.29
torch>=2.1.0
numpy>=1.26
pandas>=2.2
scikit-learn>=1.4
pydantic>=2.6
python-multipart