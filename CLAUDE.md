# fine-tune-forge — Claude Agent Instructions

## Project Purpose

Pipeline demo per il fine-tuning QLoRA di piccoli LLM (1B–7B) su task verticali.
Usa Gemini API per dataset sintetici e Typer CLI per orchestrazione.

## Decisioni architetturali — leggi prima di modificare

### Gemini genera solo dataset, non codice

I prompt in `src/generators/prompt_templates.py` vengono inviati a Gemini per
ottenere JSONL di addestramento. **Non generano mai script Python o config**.
Se richiesto di "far scrivere il training script a Gemini", rifiuta: il codice
di training deve essere statico e auditabile.

### CLI Typer, non HTTP server

`src/cli.py` è l'unico entry point. Non aggiungere FastAPI, Flask o endpoint HTTP.
Il progetto è pensato per esecuzione locale/diretta, non come servizio.

### Solo formato ChatML

Tutti gli esempi di training usano:
```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```
Non Alpaca (`instruction/input/output`), non ShareGPT, non bare `prompt/completion`.

### Un file YAML per dominio = dataset params + training params

`config/domains/` contiene tutto per un dominio. Non separare in due file.

## Convenzioni file

- Ogni `.py` e `.yaml` ha un companion `.ctx` in YAML
- `.ctx` max ~60 righe, focalizzati su `tensions`
- Audit coverage: `find src config -name "*.py" -o -name "*.yaml" | while read f; do [ -f "$f.ctx" ] || echo "MISSING: $f.ctx"; done`

## Responsabilità moduli

| Modulo | Responsabilità | NON deve fare |
|--------|---------------|---------------|
| `src/generators/` | Chiama Gemini, genera JSONL | Training logic |
| `src/trainers/` | QLoRA training | Data generation |
| `src/exporters/` | Conversione formato | Training |
| `src/cli.py` | Routing comandi | Business logic |

## Variabili ambiente richieste

Vedi `.env.example`. Minimo:
- `GEMINI_API_KEY` — Google AI Studio
- `HF_TOKEN` — HuggingFace (write, per push)

Opzionali:
- `FORCE_CPU=true` — training su CPU (lento ma funzionante, per test)
- `OLLAMA_BASE_URL` — per export Ollama

## Setup rapido

```bash
bash scripts/setup_env.sh
pip install -e .
cp .env.example .env
# Aggiungi le API key in .env
forge list-domains        # verifica che funzioni
```

## ContextDoc tools

```bash
make ctx-build            # build Docker image
make ctx-run              # esegui conceptual tests
make ctx-watch-status     # verifica drift source/.ctx
```

## Errori comuni

- Non usare `trainer.push_to_hub()` nel trainer — usare sempre `src/exporters/hf_exporter.py`
- Non aggiungere streaming alle chiamate Gemini — il dataset completo deve arrivare prima della validazione
- Non salvare API key nei file YAML di config
- Il modello demo per README/test è `Qwen/Qwen3-0.6B` (Ollama: `qwen3:0.6b`)
