FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts

# CPU-only torch + torchvision from the same index — avoids pulling several GB
# of unused CUDA/NVIDIA packages (we only run Docling on CPU), and avoids an
# ABI mismatch from installing torch and torchvision from different sources.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV HF_HOME=/app/.cache/huggingface

# Pre-download sentence-transformers model at build time so the embed-worker
# starts instantly without a runtime download (~90 MB, cached in image layer).
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-MiniLM-L6-v2')"

# No fixed script — the ECS task `command` selects which worker runs, e.g.:
#   ["python", "scripts/process_text_worker.py", "--limit", "20"]
#   ["python", "scripts/process_table_worker.py", "--limit", "5"]
#   ["python", "scripts/process_ocr_worker.py", "--limit", "5"]
#   ["python", "scripts/process_chunk_worker.py", "--limit", "20"]
#   ["python", "scripts/process_embed_worker.py", "--loop"]
ENTRYPOINT ["python"]
