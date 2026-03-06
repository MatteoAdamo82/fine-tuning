FROM python:3.11-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends git ca-certificates build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir .

COPY . .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["forge"]
CMD ["--help"]
