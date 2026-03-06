"""
Parametric prompt templates for Gemini dataset generation.

Builds the (system_prompt, user_prompt) pair sent to Gemini to generate
synthetic ChatML training examples for a given domain.
"""
from __future__ import annotations


def build_dataset_prompt(
    obiettivo: str,
    lingua: str,
    dominio: str,
    n_examples: int,
    wrap_in_object: bool = False,
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for the LLM API call.

    The system prompt sets quality constraints and output format.
    The user prompt provides task-specific parameters.

    Args:
        wrap_in_object: if True, asks for {"examples": [...]} instead of a bare
            array. Required for APIs that enforce JSON object mode (Groq, Ollama).
    """
    if wrap_in_object:
        format_instruction = (
            'Rispondi SOLO con un JSON object valido, senza markdown, senza spiegazioni:\n'
            '{{\n'
            '  "examples": [\n'
            '    {{\n'
            '      "messages": [\n'
            '        {{"role": "system", "content": "..."}},\n'
            '        {{"role": "user", "content": "..."}},\n'
            '        {{"role": "assistant", "content": "..."}}\n'
            '      ]\n'
            '    }}\n'
            '  ]\n'
            '}}'
        )
    else:
        format_instruction = (
            'Rispondi SOLO con un JSON array valido, senza markdown, senza spiegazioni:\n'
            '[\n'
            '  {{\n'
            '    "messages": [\n'
            '      {{"role": "system", "content": "..."}},\n'
            '      {{"role": "user", "content": "..."}},\n'
            '      {{"role": "assistant", "content": "..."}}\n'
            '    ]\n'
            '  }}\n'
            ']'
        )

    system_prompt = f"""Sei un esperto generatore di dati sintetici per il fine-tuning di LLM.

Il tuo compito è generare conversazioni di addestramento realistiche e diversificate.

FORMATO OBBLIGATORIO - {format_instruction}

REGOLE QUALITÀ:
1. Ogni conversazione deve iniziare con 1 system message, poi alternare user/assistant
2. Il system message deve definire il ruolo e i limiti dell'agente in modo preciso
3. Includi casi edge: rifiuti educati, ambiguità, richieste incomplete, clienti impazienti
4. Lingua di output: {lingua}
5. NON generare conversazioni duplicate o quasi identiche
6. Le risposte dell'assistente devono essere naturali, mai robotiche
7. Includi sia conversazioni brevi (2-3 turni) che multi-turno (5-8 turni)
8. Il 100% delle conversazioni deve rispettare lo scopo del dominio: {dominio}
"""

    user_prompt = f"""Genera esattamente {n_examples} conversazioni di addestramento per questo agente:

OBIETTIVO AGENTE:
{obiettivo}

DOMINIO: {dominio}
LINGUA: {lingua}

Distribuisci le conversazioni coprendo questi scenari:
- 40% flusso principale felice (happy path, prenotazione/vendita completata)
- 20% richieste incomplete (mancano informazioni, l'agente deve chiedere)
- 20% rifiuti educati (richiesta fuori dominio, l'agente rifiuta gentilmente)
- 10% input ambigui (data/orario non chiari, l'agente chiede chiarimenti)
- 10% scenari di errore (slot non disponibile, prodotto esaurito) con proposta alternativa

IMPORTANTE: Ogni system message deve essere identico o molto simile tra tutte le conversazioni
dello stesso dominio (è il prompt di sistema del modello fine-tunato).
"""

    return system_prompt, user_prompt
