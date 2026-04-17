FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip && \
    python -m pip install -e .

EXPOSE 7860

CMD ["python", "app.py"]
