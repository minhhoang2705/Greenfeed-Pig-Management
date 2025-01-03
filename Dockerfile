# Use official Python 3.12 image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NGROK_AUTH_TOKEN="2r7lTF00g77XfqdgAVb6Dan1i4F_2qHJDfNU3HcoTxwK5n2zo"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install ngrok
RUN wget -q https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip && \
    unzip ngrok-stable-linux-amd64.zip -d /usr/local/bin && \
    rm ngrok-stable-linux-amd64.zip

# Copy requirements
COPY requirements.api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.api.txt

# Copy application code
COPY api/ ./api/
COPY modules/ ./modules/
COPY configs/ ./configs/
COPY logs/ ./logs/
COPY uploads/ ./uploads/
COPY results/ ./results/

# Expose port
EXPOSE 8000

# Create startup script
RUN echo '#!/bin/bash\n\
ngrok config add-authtoken $NGROK_AUTH_TOKEN\n\
ngrok http 8000 &\n\
uvicorn api.app:app --host 0.0.0.0 --port 8000\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the application
CMD ["/app/start.sh"]
