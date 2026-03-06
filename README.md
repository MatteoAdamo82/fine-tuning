# fine-tune-forge

Pipeline demo open-source per il fine-tuning di piccoli LLM su task verticali singoli.

Realizzato con il paradigma **[OpenDoc/ContextDoc](https://github.com/MatteoAdamo82/contextdoc)**.

## Cosa fa

Crea modelli AI ultra-specializzati (agente prenotazioni ristorante, agente vendite, agente prenotazioni professionisti) a partire da:

- **Gemini API** → genera dataset sintetico JSONL
- **QLoRA** → fine-tuna un piccolo LLM (0.6B–7B parametri) in modo efficiente
- **Exporters** → converte il modello in GGUF, Ollama o HuggingFace Hub

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  forge run   │────▶│ Gemini API   │────▶│ QLoRA Train  │────▶│    Export    │
│  (CLI input) │     │ (JSONL gen)  │     │ (HF/PEFT/TRL)│     │ GGUF/Ollama  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

## Prerequisiti

- Python 3.10+
- Docker (per i tool ContextDoc opzionali)
- [Google AI Studio API key](https://aistudio.google.com/)
- [HuggingFace token](https://huggingface.co/settings/tokens) (con write access)
- GPU opzionale (funziona anche su CPU con `FORCE_CPU=true`)

## Setup

```bash
# 1. Installa dipendenze
pip install -e .

# 2. Configura variabili ambiente
cp .env.example .env
# Modifica .env con le tue API key

# 3. Verifica installazione
forge list-domains
```

## Utilizzo

### Pipeline completa

```bash
forge run \
  --obiettivo "agente prenotazioni ristorante" \
  --modello "Qwen/Qwen3.5-0.8B" \
  --lingua italiano \
  --dominio restaurant_booking
```

### Solo generazione dataset

```bash
forge dataset --dominio restaurant_booking --output data/raw/
```

### Solo training (su dataset esistente)

```bash
forge train \
  --dataset data/processed/restaurant_booking.jsonl \
  --dominio restaurant_booking \
  --modello "Qwen/Qwen3.5-0.8B"
```

### Export modello

```bash
# GGUF (per llama.cpp / Ollama)
forge export --checkpoint outputs/checkpoints/run-xyz/ --format gguf

# Ollama (crea e registra il modello localmente)
forge export --checkpoint outputs/checkpoints/run-xyz/ --format ollama --name mio-agente

# HuggingFace Hub
forge export --checkpoint outputs/checkpoints/run-xyz/ --format hf --repo-id username/mio-agente
```

### Lista domini disponibili

```bash
forge list-domains
```

## Test senza GPU

```bash
FORCE_CPU=true forge run \
  --obiettivo "agente prenotazioni ristorante" \
  --modello "Qwen/Qwen3.5-0.8B" \
  --lingua italiano \
  --dominio restaurant_booking
```

## Modelli supportati (esempi)

| Modello HuggingFace | Ollama tag | VRAM richiesta | Note |
|---------------------|------------|----------------|------|
| Qwen/Qwen3.5-0.8B | qwen3.5:0.8b | ~2GB | Demo/test (VLM text-only) |
| Qwen/Qwen3-4B | qwen3:4b | ~6GB | Bilanciato |
| Qwen/Qwen3-8B | qwen3:8b | ~10GB | Alta qualità |

## Domini predefiniti

| Dominio | Descrizione |
|---------|-------------|
| `restaurant_booking` | Agente prenotazioni ristorante |
| `sales_agent` | Agente di vendita |
| `professional_booking` | Agente prenotazioni professionisti (dentisti, etc.) |

## ContextDoc tools

Questo progetto segue il paradigma OpenDoc: ogni file sorgente ha un companion
`.ctx` con documentazione delle decisioni architetturali.

```bash
# Esegui i conceptual test via LLM
make ctx-build && make ctx-run

# Verifica drift source/.ctx
make ctx-watch-status
```

## Struttura progetto

```
fine-tune-forge/
├── src/
│   ├── cli.py              # Entry point CLI (forge)
│   ├── generators/         # Gemini API → JSONL
│   ├── trainers/           # QLoRA training
│   └── exporters/          # GGUF / Ollama / HF
├── config/
│   ├── base.yaml           # Training defaults
│   ├── domains/            # Config per dominio
│   └── models/             # Config per modello base
└── *.ctx                   # OpenDoc companion files
```

## Licenza

MIT
