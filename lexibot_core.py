# ================== IMPORTS ==================
import os
import re
import numpy as np
import pandas as pd
import torch
import faiss
import spacy

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_grad_enabled(False)

# ================== 1. LOAD LAW DATA ==================
print("🔹 Loading law data...")

law_df = None

# Preferred path
if os.path.exists("cyber_law_sections_unique.csv"):
    law_df = pd.read_csv("cyber_law_sections_unique.csv")
# Fallback: raw super dataset
elif os.path.exists("cyber_law_super_dataset.csv"):
    raw = pd.read_csv("cyber_law_super_dataset.csv")
    cols = ["law", "section_number", "section_title", "section_text", "punishment"]
    for c in cols:
        if c not in raw.columns:
            raw[c] = ""
    law_df = raw[cols].drop_duplicates().reset_index(drop=True)
else:
    raise FileNotFoundError(
        "I could not find data/cyber_law_sections_unique.csv or cyber_law_super_dataset.csv.\n"
        "Please upload one of them to Colab."
    )

law_df = law_df.fillna("")
print("   Law rows:", len(law_df))


# ================== 2. BUILD SENTENCE-BERT + FAISS INDEX ==================
print("🔹 Loading SentenceTransformer and building FAISS index...")

embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(embed_model_name)

texts = (law_df["section_title"] + " " + law_df["section_text"]).tolist()
embeddings = embed_model.encode(texts, convert_to_numpy=True).astype("float32")

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

metadata = law_df.to_dict(orient="records")
print("   FAISS index built with", index.ntotal, "vectors.")


# ================== 3. LOAD INTENT MODEL ==================
print("🔹 Loading intent classifier from ./cyber_intent_model ...")

if not os.path.exists("cyber_intent_model"):
    raise FileNotFoundError(
        "Folder 'cyber_intent_model' not found.\n"
        "Train the intent model first (previous cell) so that this folder exists."
    )

intent_tokenizer = AutoTokenizer.from_pretrained("cyber_intent_model")
intent_model = AutoModelForSequenceClassification.from_pretrained("cyber_intent_model")
intent_model.eval()

# HuggingFace stores id2label in config
id2label = intent_model.config.id2label
label2id = intent_model.config.label2id


# ================== 4. LOAD spaCy (for light NER) ==================
print("🔹 Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")


# ================== 5. RETRIEVER FUNCTION ==================
def retrieve_laws(query: str, top_k: int = 5):
    """Search top_k closest law sections for a text query."""
    vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(vec, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        m = metadata[int(idx)]
        results.append({
            "law": m.get("law", ""),
            "section_number": m.get("section_number", ""),
            "section_title": m.get("section_title", ""),
            "section_text": m.get("section_text", ""),
            "punishment": m.get("punishment", ""),
            "score": float(dist),
        })
    return results


# ================== 6. INTENT PREDICTION ==================
def predict_intent(text: str):
    enc = intent_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        logits = intent_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    idx = int(probs.argmax())
    label = id2label.get(idx, "UNKNOWN")
    conf = float(probs[idx])
    return label, conf


# ================== 7. LIGHT ENTITY / INFO EXTRACTION ==================
PLATFORM_KEYWORDS = [
    "instagram", "facebook", "whatsapp", "telegram", "snapchat",
    "gmail", "email", "twitter", "x.com", "youtube", "linkedin",
    "paytm", "phonepe", "gpay", "google pay", "upi", "bank app",
]

MONEY_REGEX = re.compile(r"\b(\d{2,10})\b")

def extract_info(text: str):
    t = text.lower()
    platform = None
    for p in PLATFORM_KEYWORDS:
        if p in t:
            platform = p
            break

    money = None
    m = MONEY_REGEX.search(text)
    if m:
        money = m.group(1)

    # You can later extend this using spaCy NER (PERSON, GPE, etc.)
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ in ("PERSON",)]
    places = [ent.text for ent in doc.ents if ent.label_ in ("GPE","LOC")]

    return {
        "platform": platform,
        "amount": money,
        "persons": persons,
        "locations": places,
    }


# ================== 8. RESPONSE GENERATION ==================
def is_high_risk(intent: str, text: str) -> bool:
    t = text.lower()
    red_words = ["threat", "kill", "suicide", "extort", "blackmail", "murder"]
    if any(w in t for w in red_words):
        return True
    if intent in ("CYBER_HARASSMENT", "CYBER_PRIVACY_VIOLATION"):
        return True
    return False


def build_reply(user_text: str):
    # 1) intent
    intent, conf = predict_intent(user_text)

    # 2) info
    info = extract_info(user_text)
    platform = info["platform"]
    amount = info["amount"]

    # 3) retrieve laws
    query = user_text + (" " + platform if platform else "")
    laws = retrieve_laws(query, top_k=5)

    # 4) safety message
    safety_msg = ""
    if is_high_risk(intent, user_text):
        safety_msg = (
            "⚠ *Safety Notice:*\n"
            "If you feel in immediate danger, please contact your nearest police station or "
            "local emergency number right now. Your safety is the top priority.\n\n"
        )

    # 5) summary
    summary = f"🧠 *Detected intent:* {intent}  (confidence: {conf:.2f})\n"
    if platform or amount or info["locations"]:
        summary += "*Details I could pick from your message:*\n"
        if platform:
            summary += f"• Platform / service: *{platform}*\n"
        if amount:
            summary += f"• Approx. amount involved: *₹{amount}*\n"
        if info["locations"]:
            summary += f"• Possible place / city: *{', '.join(info['locations'])}*\n"
        summary += "\n"

    # 6) law block
    if laws:
        law_block = "📜 *Relevant legal sections (approximate match):*\n"
        for l in laws[:3]:
            law_block += f"- *{l['law']} Section {l['section_number']} — {l['section_title']}*\n"
        law_block += "\n"
    else:
        law_block = "📜 I could not confidently match any specific section, but this still looks like a cyber issue.\n\n"

    # 7) guidance by intent
    if intent == "CYBER_HACKING":
        guidance = (
            "🛠 *Suggested steps for hacking / account compromise:*\n"
            "1. Change your password immediately and enable 2FA.\n"
            "2. Log out from all devices and revoke unknown sessions.\n"
            "3. Use the platform's help / support to report account compromise.\n"
            "4. If money is involved, contact your bank / UPI provider and freeze accounts.\n"
            "5. File a complaint on the National Cyber Crime Portal: https://cybercrime.gov.in/\n"
        )
    elif intent == "CYBER_FRAUD":
        guidance = (
            "🛠 *Suggested steps for UPI / online payment fraud:*\n"
            "1. Immediately call your bank's helpline and report unauthorized transactions.\n"
            "2. Freeze or temporarily block your card / UPI / netbanking.\n"
            "3. Collect evidence: SMS, transaction IDs, screenshots of chats / apps.\n"
            "4. File a complaint at https://cybercrime.gov.in/ with full details.\n"
        )
    elif intent == "CYBER_IDENTITY_THEFT":
        guidance = (
            "🛠 *Suggested steps for identity theft / fake profile:*\n"
            "1. Report the fake profile / impersonation on the platform.\n"
            "2. Inform friends / contacts not to trust messages from that account.\n"
            "3. Keep screenshots as evidence (profile URL, chats, etc.).\n"
            "4. File a cyber complaint if the misuse is serious (fraud, defamation, etc.).\n"
        )
    elif intent == "CYBER_HARASSMENT":
        guidance = (
            "🛠 *Suggested steps for cyber harassment / stalking:*\n"
            "1. Block the abuser; avoid replying to them.\n"
            "2. Keep screenshots of offensive messages, calls logs, etc.\n"
            "3. Use in-app reporting tools to report harassment.\n"
            "4. File a complaint on the cybercrime portal and, if threats are severe, at the local police station.\n"
        )
    elif intent == "CYBER_PRIVACY_VIOLATION":
        guidance = (
            "🛠 *Suggested steps for privacy violations (leaked photos / data):*\n"
            "1. Capture screenshots / URLs of the leaked content.\n"
            "2. Report content to the platform asking for urgent takedown.\n"
            "3. If intimate / highly sensitive content is shared, file a complaint immediately.\n"
            "4. You can use https://cybercrime.gov.in/ (especially the 'Women / Child' section).\n"
        )
    else:
        guidance = (
            "🛠 *General steps for cyber incidents:*\n"
            "1. Preserve all evidence: screenshots, chat logs, emails, transaction IDs.\n"
            "2. Report the incident on the relevant platform (Instagram, bank app, etc.).\n"
            "3. File a complaint on the National Cyber Crime Portal: https://cybercrime.gov.in/\n"
        )

    return safety_msg + summary + law_block + guidance


# ================== 9. CHAT LOOP ==================
print("\n✅ Cyber LexiBot is ready! Type your cyber law problem.")
print("   Type 'quit' to exit.\n")

while True:
    try:
        user = input("You: ").strip()
    except EOFError:
        break

    if not user:
        continue
    if user.lower() in ("quit", "exit", "bye"):
        print("LexiBot: Take care and stay safe online! 👋")
        break

    reply = build_reply(user)
    print("\nLexiBot:\n", reply, "\n")