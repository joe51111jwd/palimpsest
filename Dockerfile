# Palimpsest memory service — the Add/Search contract used by the
# Agent Memory Leaderboard (https://agentmemories.ai).
#
#   docker build -t palimpsest .
#   docker run -p 8000:8000 palimpsest
#
# Then:
#   GET  /health
#   POST /add     {request_id, messages:[{role, content, timestamp?}], user_id, session_id}
#   POST /search  {query, user_id, top_k}
#
# The image is self-contained and needs no network at runtime: the embedding
# weights are baked in at build time, and retrieval uses no LLM at all.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY pyproject.toml README.md ./
COPY palimpsest ./palimpsest

RUN pip install --no-cache-dir -e . \
 && pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.27"

# Bake the embedding model into the image so the container never reaches the
# network at runtime, and so a cold start does not pay a download.
RUN python -c "from palimpsest.embed import default_embedder; print(default_embedder().backend)" \
 && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Extraction mode:
#   llm  — full engine: the claim ledger is populated by an LLM extractor on
#          write. Needs a reachable model; set PALIMPSEST_MODEL and whatever
#          credentials your client requires.
#   none — no LLM anywhere. The episodic index still serves retrieval, but the
#          interval ledger stays empty, which removes the supersession and
#          as-of behaviour that distinguishes this system. Weaker, and labelled
#          weaker rather than passed off as the real thing.
ENV PALIMPSEST_EXTRACTOR=none

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status < 300 else 1)"

CMD ["uvicorn", "palimpsest.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
