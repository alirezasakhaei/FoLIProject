FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Install uv into a directory already on PATH
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies using uv
RUN sed -i '/--extra-index-url/d' requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the entire project
COPY . .

# Fix line endings for scripts (in case built on Windows)
RUN apt-get update && apt-get install -y dos2unix && \
    find /workspace -type f -name "*.sh" -exec dos2unix {} \; && \
    apt-get remove -y dos2unix && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/checkpoints /workspace/results /workspace/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WANDB_DIR=/workspace/results

# Make scripts executable
RUN chmod +x /workspace/run_experiment.sh /workspace/recipes/run_all_recipes.sh

# Default command (can be overridden for interactive sessions)
CMD ["/bin/bash"]
