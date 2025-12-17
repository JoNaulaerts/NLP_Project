FROM python:3.11-slim

WORKDIR /app

# System deps + Docker CLI (for docker socket access patterns)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg apt-transport-https \
  && mkdir -p /etc/apt/keyrings \
  && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
  && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" \
     > /etc/apt/sources.list.d/docker.list \
  && apt-get update \
  && apt-get install -y --no-install-recommends docker-ce-cli \
  && rm -rf /var/lib/apt/lists/*

# Install docker-mcp CLI plugin (so `docker mcp ...` exists)
# ARG MCP_VERSION=v0.7.0
# RUN mkdir -p /root/.docker/cli-plugins \
#   && curl -fsSL -o /root/.docker/cli-plugins/docker-mcp \
#      "https://github.com/docker/mcp-gateway/releases/download/${MCP_VERSION}/docker-mcp-linux-amd64" \
#   && chmod +x /root/.docker/cli-plugins/docker-mcp

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "app_new.py", "--server.address=0.0.0.0", "--server.port=8501"]
