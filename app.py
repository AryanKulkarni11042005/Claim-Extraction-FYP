import streamlit as st
import requests
import json
import re
import spacy

# ==========================================
# STREAMLIT UI SETUP & CACHING
# ==========================================
st.set_page_config(page_title="Claimify Extractor", page_icon="🔍", layout="centered")

@st.cache_resource
def load_nlp_model():
    """Loads the spaCy model. Downloads it automatically if missing."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        import sys
        st.warning("Downloading spaCy 'en_core_web_sm' model... This will only happen once.")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

ENDPOINT = "http://localhost:11434/api/chat"

# ==========================================
# HELPER FUNCTIONS (Updated to accept model_name)
# ==========================================
def call_model(prompt, model_name):
    """Sends a POST request to local Ollama and returns the assistant's message."""
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": False
    }
    
    try:
        response = requests.post(ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        content = response.json().get('message', {}).get('content', '')
        print(f"  [DEBUG] Raw Model Output:\n  {content.strip()}\n")
        return content
    except Exception as e:
        print(f"  [ERROR] API Call failed: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"  [ERROR] Response text: {response.text}")
        return ""

def safe_json_loads(text, default, original_prompt=None, model_name=None):
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
    if original_prompt and model_name:
        print("  [DEBUG] JSON parsing failed, retrying...")
        retry_prompt = f"{original_prompt}\n\nYour previous response was not valid JSON. Respond ONLY with valid JSON."
        retry_text = call_model(retry_prompt, model_name)
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
    
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            for ent in doc.ents:
                if ent.start <= token.i < ent.end:
                    entities["main_subject"] = ent.text
                    break
            
            if not entities["main_subject"] and token.pos_ in ("PROPN", "NOUN"):
                entities["main_subject"] = token.text
                
            if entities["main_subject"]:
                break
                
    return entities

def simple_reference_resolution(sentence, context_text, nlp_model):
    """Resolves references using generalized dependency and entity rules."""
    if not context_text.strip():
        return sentence

    context_data = extract_context_entities(context_text, nlp_model)
    main_subject = context_data["main_subject"]
    recent_org = context_data["recent_org"]
    recent_person = context_data["recent_person"]

    resolved = sentence

    if main_subject:
        resolved = re.sub(r'^([Ii]t|[Tt]hey|[Tt]he company|[Tt]he organization)\b', main_subject, resolved)
    elif recent_org:
        resolved = re.sub(r'^([Ii]t|[Tt]hey|[Tt]he company|[Tt]he organization)\b', recent_org, resolved)

    if recent_person:
        resolved = re.sub(r'\b([Hh]e|[Ss]he)\b', recent_person, resolved)
        resolved = re.sub(r'^([Hh]e|[Ss]he)\b', recent_person, resolved)

    if recent_org and main_subject and recent_org != main_subject:
        resolved = re.sub(r'\b(?:its|their) (technology|software|founder)\b', f"{recent_org} \\1", resolved)
        resolved = re.sub(r'\b(?:its|their) (products|services|offerings)\b', f"{main_subject} \\1", resolved)

    if main_subject:
        resolved = re.sub(r'\b(?:its|their)\b', f"{main_subject}'s", resolved)
        resolved = re.sub(r'\b[Tt]he company\b', main_subject, resolved)
        
    return resolved

# ==========================================
# LLM PROMPT STAGES
# ==========================================
def selection_stage(sentence, context, model_name):
    prompt = f"""Task: Selection
You are a factual claim selector.
You will receive:

Context: nearby sentences
Sentence: the current sentence only

Your job:
Determine whether the Sentence contains a specific and verifiable factual proposition.
Use Context only to understand ambiguous references.
Never copy facts from Context unless they are explicitly referred to in the Sentence.
Reject:
opinions
praise
rankings
speculation
introductions
conclusions
statements about future possibilities
If the Sentence contains both factual and non-factual content, keep only the factual part.

Examples:
Sentence:
"Google was founded in 1998."
Output:
{{"has_claim": true, "clean_sentence": "Google was founded in 1998."}}

Sentence:
"It is one of the best companies in the world."
Output:
{{"has_claim": false, "clean_sentence": ""}}

Sentence:
"The company later used its technology in several products."
Context:
"Apple acquired Beats Electronics in 2014 for $3 billion."
Output:
{{"has_claim": true, "clean_sentence": "The company later used its technology in several products."}}

Sentence:
"Some experts estimate inflation could rise to 300%."
Output:
{{"has_claim": false, "clean_sentence": ""}}

Sentence:
"{sentence}"
Context:
"{context}"

Return ONLY JSON:
{{"has_claim": true, "clean_sentence": "..."}}"""
    
    response = call_model(prompt, model_name)
    return safe_json_loads(response, {"has_claim": False, "clean_sentence": sentence}, prompt, model_name)

def disambiguation_stage(sentence, context, model_name):
    prompt = f"""Task: Disambiguation
Context: {context}
Sentence: {sentence}

Instructions:
- Resolve vague references naturally using the context. Reject unresolved ambiguity.
- If the sentence already contains explicit named entities, keep them unchanged.
- Never replace a named entity with a more vague phrase such as: "the company", "the organization", "it", "they".
- Prefer the main subject from the previous sentence for sentence-initial references like: "It", "They", "The company".
- Keep the resolved sentence in natural English.

Output ONLY a JSON object in this format:
{{"can_disambiguate": true, "resolved_sentence": "..."}}"""
    
    response = call_model(prompt, model_name)
    return safe_json_loads(response, {"can_disambiguate": False, "resolved_sentence": sentence}, prompt, model_name)

def decomposition_stage(sentence, model_name):
    prompt = f"""Task: Decomposition
Sentence: {sentence}

Instructions: Split the sentence into standalone factual claims. Preserve all dates, numbers, money values, and names. 
- Never use pronouns in final claims.
- Never use vague references such as "the contract", "the company", "it", "they", "this", or "that".
- All claims MUST be derived SOLELY from the provided 'Sentence'. Do NOT introduce information from previous sentences or external knowledge.
- Every claim must be understandable in complete isolation.
- Repeat the relevant subject or entity if needed.
- Do not produce multiple claims that express the same fact in different wording.
- Prefer one clear claim in active voice.
- Preserve the original wording and order when possible.
- If two possible claims are semantically identical, return only the simpler one.
- Preserve all resolved named entities exactly as they appear in the input sentence.
- Never replace "Microsoft" with "the company".
- Never replace "Apple" with "the organization".
- Never introduce new vague references.
- If the input sentence is already a single clear factual claim, return exactly one claim with the same wording.

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
"Microsoft later integrated GitHub Copilot into several of Microsoft's products."
Correct output:
{{
  "claims": [
    "Microsoft later integrated GitHub Copilot into several of Microsoft's products."
  ]
}}

Incorrect output:
{{
  "claims": [
    "Company later integrated GitHub Copilot into several of the company's products."
  ]
}}

Output ONLY a JSON object in this format:
{{"claims": ["...", "..."]}}"""
    
    response = call_model(prompt, model_name)
    return safe_json_loads(response, {"claims": []}, prompt, model_name)

# ==========================================
# MAIN PIPELINE
# ==========================================
def extract_claims(text, model_name, progress_bar=None, status_text=None):
    doc = nlp(text.strip())
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    final_claims = []
    total_sentences = len(sentences)

    for idx, sentence in enumerate(sentences):
        # Update Streamlit UI Progress
        if progress_bar and status_text:
            progress_bar.progress((idx) / total_sentences)
            status_text.write(f"**Processing sentence {idx+1}/{total_sentences}:** _{sentence}_")

        print(f"\n{'='*60}\nProcessing Sentence {idx+1}/{total_sentences}: {sentence}")
        context = get_context(sentences, idx)

        # --- STAGE 1: SELECTION ---
        selection_result = selection_stage(sentence, context, model_name)
        clean_sentence = selection_result.get("clean_sentence", "")
        
        # Safeguards
        if not clean_sentence or clean_sentence.strip() == "":
            selection_result["has_claim"] = False

        if clean_sentence and context:
            if clean_sentence.strip() in context:
                selection_result["has_claim"] = False
                selection_result["clean_sentence"] = ""
                clean_sentence = ""
            
        clean_lower = clean_sentence.lower()
        if any(pattern in clean_lower for pattern in NON_FACTUAL_PATTERNS):
            selection_result["has_claim"] = False
            selection_result["clean_sentence"] = ""
            clean_sentence = ""

        if not selection_result.get("has_claim", False):
            continue
            
        clean_sentence = selection_result.get("clean_sentence", sentence)

        # --- PRE-STAGE: SIMPLE REFERENCE RESOLUTION ---
        resolved_sentence = simple_reference_resolution(clean_sentence, context, nlp)

        # --- STAGE 2: DISAMBIGUATION ---
        if needs_disambiguation(resolved_sentence):
            disambig_result = disambiguation_stage(resolved_sentence, context, model_name)
            if not disambig_result.get("can_disambiguate", False):
                continue
            working_sentence = disambig_result.get("resolved_sentence", resolved_sentence)
        else:
            working_sentence = resolved_sentence

        # --- STAGE 3: DECOMPOSITION ---
        decomp_result = decomposition_stage(working_sentence, model_name)

        raw_claims = decomp_result.get("claims", [])
        unique_claims = []
        seen = set()

        for claim in raw_claims:
            normalized = claim.lower().strip().replace(".", "")
            if normalized not in seen:
                seen.add(normalized)
                unique_claims.append(claim)

        vague_terms_to_reject = [" the company", " the organization", " it", " they", " this", " that", " the contract"]

        filtered_claims_s4 = []
        for claim in unique_claims:
            is_vague_introduction = False
            claim_lower = claim.lower()
            working_sentence_lower = working_sentence.lower()

            for term in vague_terms_to_reject:
                if term in claim_lower and term not in working_sentence_lower:
                    is_vague_introduction = True
                    break

            if not is_vague_introduction:
                filtered_claims_s4.append(claim)

        final_verified_claims = []
        for claim in filtered_claims_s4:
            claim_doc = nlp(claim)
            working_sentence_doc = nlp(working_sentence)

            claim_key_elements = {ent.text.lower() for ent in claim_doc.ents if ent.label_ in ["ORG", "PERSON", "DATE", "GPE", "NORP", "PRODUCT", "EVENT", "LOC"]}.union({token.text for token in claim_doc if token.like_num})
            working_sentence_key_elements = {ent.text.lower() for ent in working_sentence_doc.ents if ent.label_ in ["ORG", "PERSON", "DATE", "GPE", "NORP", "PRODUCT", "EVENT", "LOC"]}.union({token.text for token in working_sentence_doc if token.like_num})

            if claim_key_elements.issubset(working_sentence_key_elements):
                final_verified_claims.append(claim)

        for claim in final_verified_claims:
            final_claims.append({
                "source_sentence": sentence,
                "claim": claim
            })

    # Complete progress bar
    if progress_bar and status_text:
        progress_bar.progress(1.0)
        status_text.success("Processing Complete!")

    return final_claims

# ==========================================
# STREAMLIT UI RENDER
# ==========================================
st.title("Claimify: Factual Claim Extractor")
st.markdown("Extract standalone, decontextualized factual claims from text using LLM pipelines.")

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox(
        "Select Model",
        options=["qwen2.5:3b", "qwen2.5:7b"],
        index=0,
        help="Make sure the selected model is running locally in Ollama."
    )
    st.divider()
    st.markdown("### Sample Texts")
    if st.button("Load Simple Text"):
        st.session_state.text_input = "Amazon was founded by Jeff Bezos in 1994. The company is headquartered in Seattle, Washington."
    if st.button("Load Medium Text"):
        st.session_state.text_input = "Tesla released the Model 3 in 2017. It quickly became the most incredible electric car on the market. Elon Musk stated that the company might produce a fully autonomous vehicle by next year. However, some analysts think this is highly unlikely."
    if st.button("Load Hard Text"):
        st.session_state.text_input = "In late 2022, OpenAI launched ChatGPT to the public. The organization hopes it will completely revolutionize the tech industry. Microsoft quickly invested $10 billion into the startup. They later integrated its underlying technology into Bing. Although many users believe it is the smartest AI ever created, regulators warned that the company could face severe antitrust scrutiny soon."

# Manage default state for text area
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

user_input = st.text_area("Paste your text here:", value=st.session_state.text_input, height=200)

if st.button("Extract Claims", type="primary"):
    if not user_input.strip():
        st.error("Please enter some text to analyze.")
    else:
        st.divider()
        st.subheader("Extraction Progress")
        
        # Setup UI elements for tracking progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run Extraction
        results = extract_claims(
            text=user_input, 
            model_name=selected_model, 
            progress_bar=progress_bar, 
            status_text=status_text
        )
        
        st.divider()
        st.subheader("Extracted Factual Claims")
        
        if not results:
            st.info("No verifiable factual claims were found in the provided text.")
        else:
            for item in results:
                with st.container():
                    st.markdown(f"**Source:** _{item['source_sentence']}_")
                    st.success(f"**Claim:** {item['claim']}")
                    st.write("") # Spacing