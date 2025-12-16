FROM python:3.11-slim

WORKDIR /app

# System deps + Node.js (for npx/MCP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg \
    docker.io \
  && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y nodejs \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt

# Install torch with CUDA 12.1 support (matching your venv)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Then install rest of requirements
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your project
COPY . /app

EXPOSE 8501

# Run Streamlit UI
CMD ["streamlit", "run", "app_new.py", "--server.address=0.0.0.0", "--server.port=8501"]
