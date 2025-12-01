#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Backend (FastAPI)..."
# Run FastAPI in the background
# We bind to 0.0.0.0:8000 so it's accessible inside the container
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to start up
echo "Waiting 5 seconds for backend to initialize..."
sleep 5

echo "Starting Frontend (Streamlit)..."
# Set the backend URL to localhost since both run in the same container
export BACKEND_URL="http://127.0.0.1:8000"

# Run Streamlit
# Render provides the PORT environment variable. Streamlit listens on port 8501 by default.
# We configure streamlit to listen on $PORT (or 8501 if not set)
PORT=${PORT:-8501}
echo "Streamlit will listen on port $PORT"

streamlit run frontend/app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
