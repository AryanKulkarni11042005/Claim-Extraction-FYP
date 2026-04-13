import json
import requests
import spacy

# -----------------------------
# Setup
# -----------------------------
nlp = spacy.load("en_core_web_sm")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


# -----------------------------
# Helper: Call local model
# -----------------------------
def call_model(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            # "format": "json"
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()["response"]


# -----------------------------
# Helper: Safe JSON parse
# -----------------------------
def safe_json_loads(text, default):
    try:
        return json.loads(text)
    except Exception as e:
        print("\nJSON PARSE ERROR:")
        print(e)
        print("RAW:")
        print(text)
        return default


# -----------------------------
# Sentence Splitter
# -----------------------------
def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# -----------------------------
# Stage 1: Selection
# Claimify-like: keep factual part only
# -----------------------------
def selection_stage(sentence, context=""):
    prompt = f"""
You are the Selection stage of a factual claim extraction system.

Your job:
1. Determine whether the sentence contains any factual, verifiable information.
2. If the sentence contains both factual and non-factual content, keep ONLY the factual part.
3. Remove:
   - opinions
   - predictions
   - speculation
   - emotional language
   - recommendations
   - vague phrases

Reject claims involving words like:
could, may, might, potentially, perhaps, likely, expected,
predicted, estimate, experts believe, severe, rampant

Return ONLY valid JSON.

Sentence:
{sentence}

Context:
{context}

Examples:

Sentence:
Tesla was founded in 2003 and is the best EV company.

Output:
{{
  "has_claim": true,
  "clean_sentence": "Tesla was founded in 2003."
}}

Sentence:
Some experts estimate inflation could rise to 300%.

Output:
{{
  "has_claim": false,
  "clean_sentence": ""
}}

Sentence:
Some experts estimate that the annual inflation rate could potentially double to 300%.

Output:
{{
  "has_claim": true,
  "clean_sentence": "Argentina's monthly inflation reached 25.5%."
}}

Expected JSON:
{{
  "has_claim": true,
  "clean_sentence": "..."
}}
"""

    result = call_model(prompt)

    print("\n====================")
    print("SELECTION INPUT:")
    print(sentence)
    print("SELECTION OUTPUT:")
    print(result)

    return safe_json_loads(
        result,
        {
            "has_claim": False,
            "clean_sentence": ""
        }
    )


# -----------------------------
# Stage 2: Disambiguation
# Claimify-style decontextualization
# -----------------------------
def disambiguation_stage(sentence, context=""):
    prompt = f"""
You are the disambiguation stage.

If the sentence is already clear and understandable on its own,
return it unchanged.

Only return can_disambiguate = false if the sentence contains unresolved references such as:
- he
- she
- they
- it
- this
- that
- the company
- the policy
- last year
- next year

Sentence:
{sentence}

Context:
{context}

Return ONLY valid JSON.

Example 1:
{{
  "can_disambiguate": true,
  "resolved_sentence": "Argentina's monthly inflation reached 25.5%."
}}

Example 2:
{{
  "can_disambiguate": false,
  "resolved_sentence": ""
}}
"""

    result = call_model(prompt)

    print("\n====================")
    print("DISAMBIGUATION INPUT:")
    print(sentence)
    print("DISAMBIGUATION OUTPUT:")
    print(result)

    return safe_json_loads(
        result,
        {
            "can_disambiguate": True,
            "resolved_sentence": sentence
        }
    )
# -----------------------------
# Stage 3: Decomposition
# Split into atomic claims
# -----------------------------
def decomposition_stage(sentence):
    prompt = f"""
You are the Decomposition stage of a factual claim extraction system.

Task:
Break the sentence into small, standalone, factual claims.

Rules:
- Each claim must be independently verifiable.
- Each claim must be a complete sentence.
- Do not repeat information.
- Return ONLY valid JSON.

Sentence:
{sentence}

Example:

Sentence:
Google was founded in 1998 by Larry Page and Sergey Brin.

Output:
{{
  "claims": [
    "Google was founded in 1998.",
    "Larry Page founded Google.",
    "Sergey Brin founded Google."
  ]
}}
"""

    result = call_model(prompt)

    print("\n====================")
    print("DECOMPOSITION INPUT:")
    print(sentence)
    print("DECOMPOSITION OUTPUT:")
    print(result)

    data = safe_json_loads(result, {"claims": []})

    if isinstance(data, dict):
        return data.get("claims", [])

    return []


# -----------------------------
# Main Pipeline
# -----------------------------
def extract_claims(text):
    sentences = split_sentences(text)

    print("\nSPLIT SENTENCES:")
    for s in sentences:
        print("-", s)

    final_claims = []

    for sentence in sentences:
        # Stage 1
        selected = selection_stage(sentence, context=text)

        clean_sentence_lower = selected.get("clean_sentence", "").lower()

        bad_words = [
            "could", "might", "may", "potentially",
            "estimate", "predict", "predicted", "likely"
        ]

        if any(word in clean_sentence_lower for word in bad_words):
            selected["has_claim"] = False
            selected["clean_sentence"] = ""

        if not selected.get("has_claim", False):
            print("-> Rejected at Selection")
            continue

        clean_sentence = selected.get("clean_sentence", "").strip()

        if not clean_sentence:
            print("-> Empty after Selection")
            continue

        # Stage 2
        clarified = disambiguation_stage(clean_sentence, context=text)

        if not clarified.get("can_disambiguate", False):
            print("-> Rejected at Disambiguation")
            continue

        resolved_sentence = clarified.get("resolved_sentence", "").strip()

        if not resolved_sentence:
            print("-> Empty after Disambiguation")
            continue

        # Stage 3
        claims = decomposition_stage(resolved_sentence)

        if not claims:
            print("-> No claims produced")
            continue

        for claim in claims:
            final_claims.append({
                "source_sentence": sentence,
                "claim": claim
            })

    return final_claims


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    text = """
Argentina's rampant inflation, with monthly rates reaching as high as 25.5%, has made many goods unobtainable and plunged the value of the currency, causing severe economic hardship.

Some experts estimate that the annual inflation rate could potentially double to 300%, while others predict even higher rates.
"""

    claims = extract_claims(text)

    print("\n====================")
    print("FINAL CLAIMS:")
    print(json.dumps(claims, indent=2))