import requests
import json
import re
import spacy
import os 
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = "openai/gpt-oss-120b:free"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: The 'en_core_web_sm' model is missing.")
    print("Please run: python -m spacy download en_core_web_sm")
    exit(1)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def call_model(prompt):
    """Sends a POST request to OpenRouter and returns the assistant's message."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    
    try:
        response = requests.post(ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        print(f"  [DEBUG] Raw Model Output:\n  {content.strip()}\n")
        return content
    except Exception as e:
        print(f"  [ERROR] API Call failed: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"  [ERROR] Response text: {response.text}")
        return ""

def safe_json_loads(text, default, original_prompt=None):
    """Extracts the first JSON object from text using regex, with a 1-retry fallback."""
    def extract(t):
        match = re.search(r'\{[\s\S]*\}', t)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    result = extract(text)
    if result is not None:
        return result

    # Retry logic
    if original_prompt:
        print("  [DEBUG] JSON parsing failed, retrying...")
        retry_prompt = f"{original_prompt}\n\nYour previous response was not valid JSON. Respond ONLY with valid JSON."
        retry_text = call_model(retry_prompt)
        retry_result = extract(retry_text)
        if retry_result is not None:
            return retry_result

    print("  [WARNING] Failed to parse JSON after retry. Returning default.")
    return default

def get_context(sentences, idx, window=2):
    """Returns the previous `window` sentences as context."""
    start = max(0, idx - window)
    return " ".join(sentences[start:idx])

# ==========================================
# RULES & PATTERNS
# ==========================================
AMBIGUOUS_WORDS = [
    " he ", " she ", " they ", " it ", " this ", " that ", 
    " these ", " their ", " its ", 
    " the company ", " the organization ", " the product ", 
    " the project ", " the policy ", " the decision ", 
    " last year ", " next year "
]

NON_FACTUAL_PATTERNS = [
    "would ", "could ", "might ", "may ", "expected to ", 
    "planned to ", "intended to ", "said it would ", 
    "said they would ", "hopes to ", "aims to ", "will help "
]

def needs_disambiguation(sentence):
    """Detects if a sentence contains known ambiguous trigger words."""
    s_lower = f" {sentence.lower()} " 
    return any(word in s_lower for word in AMBIGUOUS_WORDS)

def extract_context_entities(context_text, nlp_model):
    """Extracts the main subject and recent entities using dependency parsing."""
    doc = nlp_model(context_text)
    
    entities = {
        "main_subject": None,
        "recent_org": None,
        "recent_person": None,
        "recent_product": None
    }
    
    if not context_text.strip():
        return entities
        
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    products = [ent.text for ent in doc.ents if ent.label_ == "PRODUCT"]
    
    entities["recent_org"] = orgs[-1] if orgs else None
    entities["recent_person"] = persons[-1] if persons else None
    entities["recent_product"] = products[-1] if products else None
    
    # Find main grammatical subject using dependency parsing
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            # Check if this token belongs to a named entity
            for ent in doc.ents:
                if ent.start <= token.i < ent.end:
                    entities["main_subject"] = ent.text
                    break
            
            # Fallback to the token's text if no named entity matches (and it's not a pronoun)
            if not entities["main_subject"] and token.pos_ in ("PROPN", "NOUN"):
                entities["main_subject"] = token.text
                
            if entities["main_subject"]:
                break
                
    return entities

def simple_reference_resolution(sentence, context_text, nlp_model):
    """Resolves references using generalized dependency and entity rules."""
    if not context_text.strip():
        return sentence

    # 1. Extract context data
    context_data = extract_context_entities(context_text, nlp_model)
    main_subject = context_data["main_subject"]
    recent_org = context_data["recent_org"]
    recent_person = context_data["recent_person"]

    resolved = sentence

    # 2. Sentence-initial references -> usually main_subject of the previous sentence
    if main_subject:
        resolved = re.sub(r'^([Ii]t|[Tt]hey|[Tt]he company|[Tt]he organization)\b', main_subject, resolved)
    elif recent_org:
        resolved = re.sub(r'^([Ii]t|[Tt]hey|[Tt]he company|[Tt]he organization)\b', recent_org, resolved)

    # 3. Person pronouns -> recent_person
    if recent_person:
        resolved = re.sub(r'\b([Hh]e|[Ss]he)\b', recent_person, resolved)
        resolved = re.sub(r'^([Hh]e|[Ss]he)\b', recent_person, resolved)

    # 4. Context-dependent Possessive rules ("its", "their")
    if recent_org and main_subject and recent_org != main_subject:
        # Tech/Founder of acquired/recent company
        resolved = re.sub(r'\b(?:its|their) (technology|software|founder)\b', f"{recent_org} \\1", resolved)
        
        # Products/Services belonging to the main acting entity
        resolved = re.sub(r'\b(?:its|their) (products|services|offerings)\b', f"{main_subject} \\1", resolved)

    # 5. Catch-all remainders
    if main_subject:
        # Any remaining "its/their" -> main_subject's
        resolved = re.sub(r'\b(?:its|their)\b', f"{main_subject}'s", resolved)
        # Any remaining "the company" anywhere in the sentence
        resolved = re.sub(r'\b[Tt]he company\b', main_subject, resolved)
        
    return resolved

# ==========================================
# LLM PROMPT STAGES
# ==========================================
def selection_stage(sentence, context):
    prompt = f"""Task: Selection
Context: {context}
Sentence: {sentence}

Instructions: 
- Keep only factual, verifiable content. Remove speculation/opinion.
- Reject statements describing intentions, promises, goals, plans, or expected future outcomes.
- Reject sentences containing phrases like: "would", "could", "might", "may", "expected to", "planned to", "intended to", "said it would", "said they would", "aims to", "hopes to", "will help".

Example:
Input sentence:
"The company said it would help developers collaborate more effectively."
Correct output:
{{
  "has_claim": false,
  "clean_sentence": ""
}}

Output ONLY a JSON object in this format:
{{"has_claim": true, "clean_sentence": "..."}}"""
    
    response = call_model(prompt)
    return safe_json_loads(response, {"has_claim": False, "clean_sentence": sentence}, prompt)

def disambiguation_stage(sentence, context):
    prompt = f"""Task: Disambiguation
Context: {context}
Sentence: {sentence}

Instructions: Resolve vague references naturally using the context. Reject unresolved ambiguity.
Output ONLY a JSON object in this format:
{{"can_disambiguate": true, "resolved_sentence": "..."}}"""
    
    response = call_model(prompt)
    return safe_json_loads(response, {"can_disambiguate": False, "resolved_sentence": sentence}, prompt)

def decomposition_stage(sentence):
    prompt = f"""Task: Decomposition
Sentence: {sentence}

Instructions: Split the sentence into standalone factual claims. Preserve all dates, numbers, money values, and names. 
- Never use pronouns in final claims.
- Never use vague references such as "the contract", "the company", "it", "they", "this", or "that".
- Every claim must be understandable in complete isolation.
- Repeat the relevant subject or entity if needed.
- Do not produce multiple claims that express the same fact in different wording.
- Prefer one clear claim in active voice.
- Preserve the original wording and order when possible.
- If two possible claims are semantically identical, return only the simpler one.

Examples:
Input sentence:
"In 2021, NASA awarded SpaceX a $2.9 billion contract to build a lunar lander."
Correct output:
{{
  "claims": [
    "In 2021, NASA awarded SpaceX a $2.9 billion contract.",
    "NASA awarded SpaceX the contract to build a lunar lander."
  ]
}}

Input sentence:
"Microsoft later integrated GitHub Copilot into several of its products."
Correct output:
{{
  "claims": [
    "Microsoft later integrated GitHub Copilot into several Microsoft products."
  ]
}}

Output ONLY a JSON object in this format:
{{"claims": ["...", "..."]}}"""
    
    response = call_model(prompt)
    return safe_json_loads(response, {"claims": []}, prompt)

# ==========================================
# MAIN PIPELINE
# ==========================================
def extract_claims(text):
    doc = nlp(text.strip())
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    final_claims = []

    for idx, sentence in enumerate(sentences):
        print(f"\n{'='*60}\nProcessing Sentence {idx+1}/{len(sentences)}: {sentence}")
        context = get_context(sentences, idx)
        print(f"Context available: '{context}'")

        # --- STAGE 1: SELECTION ---
        print("\n--- STAGE 1: SELECTION ---")
        selection_result = selection_stage(sentence, context)
        
        clean_sentence = selection_result.get("clean_sentence", "")
        
        # SAFEGUARD 1: Check for empty or whitespace-only clean_sentence
        if not clean_sentence or clean_sentence.strip() == "":
            selection_result["has_claim"] = False
            
        # SAFEGUARD 2: Pattern matching for non-factual statements
        clean_lower = clean_sentence.lower()
        if any(pattern in clean_lower for pattern in NON_FACTUAL_PATTERNS):
            print(f"  [Safeguard] Non-factual pattern detected. Rejecting claim.")
            selection_result["has_claim"] = False
            selection_result["clean_sentence"] = ""
            clean_sentence = ""
            
        print(f"  [Parsed JSON]: {selection_result}")

        if not selection_result.get("has_claim", False):
            print("  -> No verifiable claim found. Skipping to next sentence.")
            continue
            
        clean_sentence = selection_result.get("clean_sentence", sentence)

        # --- PRE-STAGE: SIMPLE REFERENCE RESOLUTION ---
        resolved_sentence = simple_reference_resolution(clean_sentence, context, nlp)
        print(f"\n  [Pre-Disambiguation Rule Check] Simple Resolution Result: {resolved_sentence}")

        # --- STAGE 2: DISAMBIGUATION ---
        print("\n--- STAGE 2: DISAMBIGUATION ---")
        if needs_disambiguation(resolved_sentence):
            print("  -> Ambiguity detected. Calling LLM for disambiguation...")
            disambig_result = disambiguation_stage(resolved_sentence, context)
            print(f"  [Parsed JSON]: {disambig_result}")

            if not disambig_result.get("can_disambiguate", False):
                print("  -> Could not disambiguate. Skipping to next sentence.")
                continue
            working_sentence = disambig_result.get("resolved_sentence", resolved_sentence)
        else:
            print("  -> No ambiguity detected. Skipping LLM disambiguation.")
            working_sentence = resolved_sentence

        # --- STAGE 3: DECOMPOSITION ---
        print("\n--- STAGE 3: DECOMPOSITION ---")
        decomp_result = decomposition_stage(working_sentence)
        print(f"  [Parsed JSON]: {decomp_result}")

        # SAFEGUARD 3: Deduplicate identically worded claims
        raw_claims = decomp_result.get("claims", [])
        unique_claims = []
        seen = set()

        for claim in raw_claims:
            normalized = claim.lower().strip().replace(".", "")
            if normalized not in seen:
                seen.add(normalized)
                unique_claims.append(claim)

        # Append unique claims to final list
        for claim in unique_claims:
            final_claims.append({
                "source_sentence": sentence,
                "claim": claim
            })

    return final_claims

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Expanded test text to ensure general rules work across different contexts
    test_text = '''
Apple acquired Beats Electronics in 2014 for $3 billion. 
The company later used its technology in several products.
Microsoft acquired GitHub in 2018. 
It later integrated GitHub Copilot into several of its products.
Elon Musk founded SpaceX in 2002. 
He later became the CEO of Tesla.
    '''
    
    print("Starting Claimify Pipeline...")
    results = extract_claims(test_text)
    
    print("\n" + "="*60)
    print("FINAL EXTRACTED CLAIMS:")
    print("="*60)
    print(json.dumps(results, indent=2))