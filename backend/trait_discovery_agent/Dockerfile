FROM python:3.12-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Add project root to Python path
ENV PYTHONPATH=/app

WORKDIR /app

# Install dependencies first (better Docker caching)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default command
CMD ["pytest", "tests/", "-v"]