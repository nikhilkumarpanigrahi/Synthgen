FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU first (saves ~3GB download & builds in seconds)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure data and cache directories have full write permissions for non-root container users
RUN mkdir -p /app/data && chmod -R 777 /app

# Expose standard port
EXPOSE 7860

# Environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Launch server
CMD ["python", "server.py"]
