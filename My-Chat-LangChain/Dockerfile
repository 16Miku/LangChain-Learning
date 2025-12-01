# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# build-essential for compiling likely needed extensions
# curl for healthchecks or testing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
# PYTHONPATH ensures backend modules can be imported correctly
ENV PYTHONPATH=/app/backend
# DATA_DIR for persistence (will be mounted to a disk in Render)
ENV DATA_DIR=/var/lib/data

# Make the entrypoint script executable
RUN chmod +x start.sh

# Expose the port Streamlit will run on (Render sets PORT env var, we default to 8501)
EXPOSE 8501

# Run the start script
CMD ["./start.sh"]
