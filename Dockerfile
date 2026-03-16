# ==========================================
# STAGE 1: Builder
# ==========================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# THE FIX: Explicitly install the CPU-only version of PyTorch first
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Then install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# STAGE 2: Production Runner
# ==========================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Create a dedicated, non-root system user for security
RUN addgroup --system appgroup && adduser --system --group appuser

WORKDIR /app

# Copy ONLY the compiled virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code and set ownership to our non-root user
COPY --chown=appuser:appgroup agent_workflow.py app.py ./

# Switch from root to the secure non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# K8s/Docker Healthcheck to ensure the UI is actually responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]