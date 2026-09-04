FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for cache efficiency
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY NST_Code/ ./NST_Code/
COPY .env.example ./

# Expose Flask default port
EXPOSE 5000

# Use gunicorn for production server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "NST_Code.app:app"]
