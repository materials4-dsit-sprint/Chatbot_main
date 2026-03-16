FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Create storage directory used by the app
RUN mkdir -p /app/storage

# Make start script executable
RUN chmod +x start.sh

# HF Spaces expects the app to listen on 7860
EXPOSE 7860

# Start the application
CMD ["./start.sh"]