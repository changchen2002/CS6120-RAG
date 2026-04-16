# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install CPU-only PyTorch first so sentence-transformers can import it reliably
RUN pip install --no-cache-dir -f https://download.pytorch.org/whl/cpu torch==2.4.0
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Set environment variables for better performance and cache persistence
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Start uvicorn and Streamlit together using a simple script
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run ui.py --server.port 8501 --server.address 0.0.0.0"]

# This says:
#     “Build me a Python image, put my FastAPI code in it, install dependencies, expose port 8000, run FastAPI.”

# This means:

#     One container runs both FastAPI + Streamlit.
#     The CMD runs both processes — background uvicorn & Streamlit.
    

#uvicorn app:app (filename:app)  means first app is taken from filename app.py 