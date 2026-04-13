import os 
import spacy 
import json 
import requests

nlp = spacy.load("en_core_web_sm")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

def call_model(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def selection_stage(sentence, context=""):
    prompt = f"""
You are the Selection stage of a factual claim extraction system.

Task:
- Determine whether the sentence contains factual, verifiable information.
- Remove opinions, predictions, recommendations, or emotional language.
- Return ONLY JSON.

Sentence:
{sentence}

Context:
{context}

Output format:
{{
  "has_claim": true/false,
  "clean_sentence": "..."
}}
"""

    result = call_model(prompt)

    try:
        return json.loads(result)
    except:
        return {
            "has_claim": False,
            "clean_sentence": ""
        }


def disambiguation_stage(sentence, context=""):
    prompt = f"""
You are the Disambiguation stage of a claim extraction system.

Task:
- Replace pronouns and vague references with explicit entities.
- Detect unresolved ambiguity.
- If ambiguity cannot be resolved confidently, return can_disambiguate = false.
- Return ONLY JSON.

Sentence:
{sentence}

Context:
{context}

Output format:
{{
  "can_disambiguate": true/false,
  "resolved_sentence": "..."
}}
"""

    result = call_model(prompt)

    try:
        return json.loads(result)
    except:
        return {
            "can_disambiguate": False,
            "resolved_sentence": ""
        }


def decomposition_stage(sentence):
    prompt = f"""
You are the Decomposition stage of a claim extraction system.

Task:
- Break the sentence into small factual claims.
- Each claim must stand alone.
- Return ONLY JSON list.

Sentence:
{sentence}
"""

    result = call_model(prompt)

    try:
        return json.loads(result)
    except:
        return []


def extract_claims(text):
    sentences = split_sentences(text)

    final_claims = []

    for sentence in sentences:
        selected = selection_stage(sentence)

        if not selected["has_claim"]:
            continue

        clarified = disambiguation_stage(selected["clean_sentence"], context=text)

        if not clarified["can_disambiguate"]:
            continue

        claims = decomposition_stage(clarified["resolved_sentence"])

        for claim in claims:
            final_claims.append({
                "source_sentence": sentence,
                "claim": claim
            })

    return final_claims

text = """
Google was founded in 1998 by Larry Page and Sergey Brin.
It is one of the most innovative companies in the world.
"""

claims = extract_claims(text)

print(json.dumps(claims, indent=2))