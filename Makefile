CTX_COMPOSE=docker compose -f docker-compose.ctx.yml --env-file .env.ctx
CTX_RUN=python /opt/contextdoc/tools/ctx-run/ctx_run.py
CTX_WATCH=python /opt/contextdoc/tools/ctx-watch/ctx_watch.py
CTX_TARGET=src
CTX_TIMEOUT=120

.PHONY: ctx-build ctx-run ctx-run-json ctx-run-verbose ctx-run-fix ctx-watch-status ctx-watch-live ctx-watch-reverse ctx-cache-clear run stop install

ctx-build:
	$(CTX_COMPOSE) build contextdoc-tools

ctx-run:
	$(CTX_COMPOSE) run --rm contextdoc-tools $(CTX_RUN) run $(CTX_TARGET)

ctx-run-json:
	$(CTX_COMPOSE) run --rm contextdoc-tools $(CTX_RUN) run $(CTX_TARGET) --output json

ctx-run-verbose:
	$(CTX_COMPOSE) run --rm contextdoc-tools $(CTX_RUN) run $(CTX_TARGET) --verbose --timeout $(CTX_TIMEOUT)

ctx-run-fix:
	$(CTX_COMPOSE) run --rm contextdoc-tools $(CTX_RUN) run $(CTX_TARGET) --fix --timeout $(CTX_TIMEOUT)

ctx-watch-status:
	$(CTX_COMPOSE) run --rm contextdoc-tools $(CTX_WATCH) status . --since 86400

ctx-watch-live:
	$(CTX_COMPOSE) run --rm contextdoc-tools $(CTX_WATCH) watch . --grace 300

ctx-watch-reverse:
	$(CTX_COMPOSE) run --rm contextdoc-tools $(CTX_WATCH) status . --reverse

ctx-cache-clear:
	$(CTX_COMPOSE) run --rm contextdoc-tools $(CTX_RUN) clear-cache

install:
	pip install -e ".[dev]"

run:
	docker compose up -d

stop:
	docker compose down
