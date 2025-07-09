# Development Environment for Trading System
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install HuggingFace Hub CLI and core dependencies
ARG HF_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}

# Copy requirements first for better caching
COPY requirements-docker.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Login to HuggingFace Hub if token provided
RUN if [ -n "${HF_TOKEN}" ]; then \
        huggingface-cli login --token ${HF_TOKEN} --add-to-git-credential; \
    fi

# Copy source code
COPY . .

# Create directories for data and logs
RUN mkdir -p data logs models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose common ports
EXPOSE 8000 8001 8002 8003 3000 9090

# Default command
CMD ["python", "-c", "print('Trading System Development Environment Ready!')"] 