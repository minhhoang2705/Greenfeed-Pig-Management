FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    supervisor \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.api.txt requirements.frontend.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.api.txt
RUN pip3 install --no-cache-dir -r requirements.frontend.txt

# Copy the application code
COPY api/ api/
COPY frontend/ frontend/
COPY modules/ modules/
COPY weights/ weights/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports
EXPOSE 8000 8501

# Start services using supervisord
CMD ["/usr/bin/supervisord"]
