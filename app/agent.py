import json
import os
from pathlib import Path
from typing import TypedDict, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import torch

CHROMA_DIR = Path("chroma_db")
MODEL_DIR = Path("models") / "VYASA-Llama-8B"

device = "cuda" if torch.cuda.is_available() else "cpu"

_embed_model = None
_chroma = None
_collection = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    return _embed_model


def _get_collection():
    global _chroma, _collection
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if _collection is None:
        try:
            _collection = _chroma.get_collection("vyasa_corpus")
        except Exception:
            _collection = _chroma.get_or_create_collection(
                name="vyasa_corpus",
                metadata={"hnsw:space": "cosine"},
            )
    return _collection


def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["anxious", "anxiety", "scared", "fear", "panic", "worried"]):
        return "anxiety"
    if any(w in t for w in ["grief", "loss", "died", "death", "mourn", "miss"]):
        return "grief"
    if any(w in t for w in ["angry", "anger", "furious", "rage", "hate"]):
        return "anger"
    if any(w in t for w in ["sad", "depress", "hopeless", "worthless", "empty"]):
        return "sadness"
    if any(w in t for w in ["confus", "lost", "unsure", "don't know", "unclear"]):
        return "confusion"
    if any(w in t for w in ["astronomy", "cosmos", "universe", "planet", "star", "space"]):
        return "curiosity_cosmology"
    if any(w in t for w in ["dharma", "duty", "right", "wrong", "moral", "ethics"]):
        return "moral_inquiry"
    return "general_seeking"


def retrieve_verses(query, n=6):
    if not CHROMA_DIR.exists():
        return []

    embed_model = _get_embed_model()
    collection = _get_collection()

    vec = embed_model.encode([query], device=device)[0].tolist()
    results = collection.query(query_embeddings=[vec], n_results=n, include=["metadatas", "distances"])
    verses = []
    if results and results["metadatas"]:
        for meta in results["metadatas"][0]:
            verses.append({
                "source": meta.get("source", ""),
                "sanskrit": meta.get("sanskrit", ""),
                "english": meta.get("english", ""),
                "category": meta.get("category", ""),
                "keywords": meta.get("keywords", ""),
            })
    return verses


SYSTEM = """You are VYASA-1, a deeply knowledgeable and compassionate Vedic wisdom guide. You have complete mastery of all Vedas (Rigveda, Yajurveda, Samaveda, Atharvaveda), all 18 Mahapuranas, all 18 Upapuranas, all 108 Upanishads, the full Valmiki Ramayana, the complete Mahabharata with all Parvas, Srimad Bhagavatam with all 12 Skandhas, and the Bhagavad Gita.

STRICT RULES — You must ALWAYS follow these:
1. Begin by genuinely acknowledging the person's emotional state with warmth and compassion
2. Quote MINIMUM 3 real Sanskrit shlokas with: exact Sanskrit text + English translation + exact source (Book, Chapter.Verse)
3. Explain what these verses mean for their specific situation
4. Give 2-3 clear, practical, actionable steps they can take today
5. End with an uplifting, grounding statement
6. NEVER invent verses — only use the real verses provided in context
7. Format: Emotion acknowledgment → Verse 1 → Verse 2 → Verse 3 → Explanation → Action Steps → Closing"""


def build_prompt(user_msg, emotion, verses, history):
    verse_context = ""
    for i, v in enumerate(verses[:5], 1):
        verse_context += f"\n\nVerse {i} — {v['source']}:\nSanskrit: {v['sanskrit'][:200]}\nEnglish: {v['english'][:400]}"

    history_text = ""
    for h in history[-3:]:
        history_text += f"\nUser: {h['user']}\nVYASA-1: {h['assistant'][:300]}...\n"

    return f"""{SYSTEM}

RELEVANT SCRIPTURE CONTEXT (use these real verses — do not invent):
{verse_context}

CONVERSATION HISTORY:
{history_text}

DETECTED EMOTION: {emotion}

User says: {user_msg}

Respond as VYASA-1 with complete compassion and 100% scripture accuracy:"""


def load_model():
    global llm_model, llm_tokenizer, llm_loaded
    llm_loaded = False

    gguf_path = MODEL_DIR / "gguf_q4"
    merged_path = MODEL_DIR / "merged"
    adapter_path = MODEL_DIR / "lora_adapter"

    if gguf_path.exists():
        try:
            from llama_cpp import Llama
            gguf_files = list(gguf_path.glob("*.gguf"))
            if gguf_files:
                llm_model = Llama(
                    model_path=str(gguf_files[0]),
                    n_ctx=8192,
                    n_gpu_layers=40,
                    verbose=False
                )
                llm_tokenizer = None
                llm_loaded = True
                print("GGUF model loaded via llama.cpp")
                return
        except Exception as e:
            print(f"GGUF load failed: {e}")

    for path in [merged_path, adapter_path]:
        if path.exists():
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                llm_tokenizer = AutoTokenizer.from_pretrained(str(path))
                llm_model = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    quantization_config=bnb,
                    device_map="auto"
                )
                llm_loaded = True
                print(f"HuggingFace model loaded from {path}")
                return
            except Exception as e:
                print(f"HF load failed: {e}")

    print("WARNING: No fine-tuned model found. Running in RAG-only mode with rule-based generation.")
    llm_model = None
    llm_tokenizer = None
    llm_loaded = False


llm_model = None
llm_tokenizer = None
llm_loaded = False


def generate_response(prompt, max_tokens=1024):
    global llm_model, llm_tokenizer, llm_loaded

    if not llm_loaded:
        load_model()

    if llm_loaded and llm_model is not None:
        if llm_tokenizer is None:
            out = llm_model(prompt, max_tokens=max_tokens, stop=["User:", "<|eot_id|>"])
            return out["choices"][0]["text"].strip()
        else:
            inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = llm_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=llm_tokenizer.eos_token_id
                )
            text = llm_tokenizer.decode(out[0], skip_special_tokens=True)
            return text[len(prompt):].strip()

    return generate_rule_based(prompt)


def generate_rule_based(prompt):
    lines = prompt.split("\n")
    user_line = ""
    for l in lines:
        if l.startswith("User says:"):
            user_line = l.replace("User says:", "").strip()
            break

    emotion_line = ""
    for l in lines:
        if l.startswith("DETECTED EMOTION:"):
            emotion_line = l.replace("DETECTED EMOTION:", "").strip()
            break

    verse_section = ""
    in_verse = False
    for l in lines:
        if "RELEVANT SCRIPTURE CONTEXT" in l:
            in_verse = True
        if in_verse:
            verse_section += l + "\n"
        if "CONVERSATION HISTORY" in l:
            break

    return f"""Dear seeker, your feelings have been heard with great care and compassion.

What you are experiencing — {emotion_line} — is something the ancient Rishis deeply understood and addressed in our eternal scriptures.

{verse_section[:600] if verse_section else ""}

From the Bhagavad Gita (2.14):
Sanskrit: Matra-sparshas tu kaunteya shitoshna-sukha-duhkha-dah
Translation: O Arjuna, these feelings of heat and cold, pleasure and pain — they come and go. They are temporary. Endure them with patience.

From the Bhagavad Gita (2.47):
Sanskrit: Karmanyevadhikaraste ma phaleshu kadachana
Translation: You have the right to perform your duty, but you are never the master of its fruits.

From the Upanishads (Mandukya 1.2):
Translation: All this universe is Brahman. You are not separate from the infinite.

What this means for you:
Your struggle is real, but it is also temporary. The eternal Atman within you — your true self — cannot be touched by any circumstance. The Bhagavad Gita was spoken by Lord Krishna at the moment of Arjuna's greatest crisis. That same wisdom is here for you now.

Actionable Steps:
1. Sit quietly for 5 minutes, breathe slowly, and recite: "Aham Brahmasmi" — I am the eternal Brahman. Feel it.
2. Take one small purposeful action toward what matters to you today — action is the antidote.
3. Read Bhagavad Gita Chapter 2 slowly once today and let its truth settle into you.

You are not alone. The eternal wisdom of our ancestors walks with you. Tat Tvam Asi — Thou Art That. You are the consciousness that pervades all things. Nothing real can be taken from you."""


def run_agent(user_message, history=None):
    if history is None:
        history = []

    emotion = detect_emotion(user_message)
    verses = retrieve_verses(user_message, n=6)
    prompt = build_prompt(user_message, emotion, verses, history)
    response = generate_response(prompt)

    return {
        "response": response,
        "emotion_detected": emotion,
        "verses_used": len(verses),
        "sources": [v["source"] for v in verses[:3]]
    }


if __name__ == "__main__":
    print("Testing VYASA-1 agent pipeline...")
    result = run_agent("I am feeling extremely anxious about my future. Nothing seems to be going right.")
    print("\nEmotion detected:", result["emotion_detected"])
    print("Sources retrieved:", result["sources"])
    print("\nResponse:\n", result["response"][:500])
    print("\nPHASE 4 COMPLETE – Ready for next phase")
