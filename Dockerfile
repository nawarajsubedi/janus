FROM python:3.12-slim
WORKDIR /srv
ENV HF_HOME=/models PIP_NO_CACHE_DIR=1
# All deps (CPU-only torch, gliner, fastapi, ...) are pre-downloaded into ./wheels, so the build is offline
COPY wheels /wheels
RUN pip install --no-index --find-links /wheels torch fastapi httpx uvicorn pydantic-settings gliner && rm -rf /wheels
COPY app app
RUN useradd -r app && mkdir /models && chown app /models
USER app
EXPOSE 8001
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8001"]
