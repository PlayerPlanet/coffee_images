# Dockerfile for Coffee Bot Poller
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.2

# Copy Poetry configuration
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not create virtual environment (we're in a container)
RUN poetry config virtualenvs.create false

# Install dependencies (only main dependencies, not dev/cpu/gpu groups)
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY coffee_images ./coffee_images

# Create directories for data and session persistence
RUN mkdir -p /app/data /app/.telethon

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    SESSION_DIR=/app/.telethon

# Run the bot poller
CMD ["python", "-m", "coffee_images.data_collection.bot_poller"]
