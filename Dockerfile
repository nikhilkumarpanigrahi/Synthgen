FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Hugging Face Spaces standard port
EXPOSE 7860

# Environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Launch server
CMD ["python", "server.py"]
