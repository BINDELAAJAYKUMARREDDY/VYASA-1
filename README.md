## VYASA-1 — Vedic Wisdom AI
**Full Name:** Vedic Wisdom Yielding Accurate Solutions & Applications

---

## WHAT THIS IS
VYASA-1 is a local AI system trained on all Vedas, 18 Mahapuranas, 18 Upapuranas, all Upanishads, Ramayana, Mahabharata, Srimad Bhagavatam, and Bhagavad Gita. It gives compassionate, scripture-accurate answers to any life problem.

---

## Step-by-step setup (VS Code, Windows)

### STEP 1 — Open the project in VS Code
1. Open VS Code
2. Click **File → Open Folder**
3. Select the `VYASA-1` folder
4. You will see all files on the left side panel

---

### STEP 2 — Open a Terminal in VS Code
1. In VS Code, click **Terminal → New Terminal** (top menu)
2. A black terminal window opens at the bottom
3. You will type all commands here

---

### STEP 3 — Create a Python virtual environment
Type this command exactly and press Enter:
```
python -m venv vyasa_env
```
Wait for it to finish (takes 10–20 seconds).

Then activate it:
```
vyasa_env\Scripts\activate
```
You should see `(vyasa_env)` appear on the left of your terminal line. This means it worked.

---

### STEP 4 — Install all dependencies
```
pip install -r requirements.txt
```
This will take 5–15 minutes. It downloads PyTorch, ChromaDB, HuggingFace, FastAPI and everything else. Let it run completely.

---

### Step 5 — Build the real corpus (downloads real datasets)
```
python scripts\build_corpus.py
```
This downloads real Vedic datasets and builds:
- `data\processed\vyasa_corpus.jsonl`
- `data\processed\vyasa_qa.jsonl`

**Time:** 10–30 minutes depending on internet speed.

You will see messages like:
```
[1/5] Loading bhavykhatri/DharmicData...
[2/5] Loading atmabodha/Vedanta_Datasets...
...
PHASE 1 COMPLETE – Ready for next phase
```

---

### Step 6 — Build the search index (ChromaDB)
```
python scripts\build_index.py
```
This builds a ChromaDB vector database so VYASA-1 can instantly search thousands of verses.

**Time:** 2–10 minutes.

You will see:
```
PHASE 2 COMPLETE – Ready for next phase
```

---

### STEP 7 (OPTIONAL but RECOMMENDED) — Run Phase 3 (Fine-tune the model)
**WARNING: This requires a Hugging Face account and Meta Llama access.**

First get Llama 3.1 access:
1. Go to https://huggingface.co and create a free account
2. Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct and request access
3. Wait for approval (usually same day)
4. Create an access token at https://huggingface.co/settings/tokens
5. Run: `huggingface-cli login` and paste your token

Then install Unsloth (fastest fine-tuning):
```
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

Then run fine-tuning:
```
python scripts/phase3_finetune.py
```

**Time:** 1–3 hours on your RTX 2000 Ada GPU.

**NOTE:** If you skip this step, VYASA-1 will still work using the RAG system (Step 5 & 6 corpus) with a built-in rule-based response engine. It will give good answers. Fine-tuning makes it even better.

---

### Step 7 — Start the VYASA-1 server
```
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

OR double-click `run_server.bat`

You will see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### Step 8 — Ask a question (API)
Open your browser and go to:
```
http://127.0.0.1:8000/ask?q=I%20feel%20anxious%20about%20my%20future
```

The golden VYASA-1 chat interface will open. Type any question or problem and receive Vedic wisdom.

---

## RUNNING INDIVIDUAL SCRIPTS AT ANY TIME

| What you want to do | Command |
|---|---|
| Rebuild corpus | `python scripts/phase1_build_corpus.py` |
| Rebuild index | `python scripts/phase2_build_index.py` |
| Fine-tune model | `python scripts/phase3_finetune.py` |
| Start server | `python app/main.py` |

---

## Project structure (strict)
```
VYASA-1/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   ├── config.py
│   │   └── logger.py
│   ├── services/
│   │   ├── chunker.py
│   │   ├── retriever.py
│   │   ├── llm.py
│   │   └── pipeline.py
│   └── models/
│       └── schema.py
├── scripts/
│   ├── download_data.py
│   ├── build_corpus.py
│   └── build_index.py
├── data/
│   ├── raw/
│   └── processed/
├── db/
│   └── chroma/
├── tests/
│   └── test_basic.py
├── requirements.txt
├── run_server.bat
└── README.md
```

## Long input support (10k–100k+ text)
VYASA-1 uses a map-reduce RAG flow:
- Split input into chunks (~800 words)
- Retrieve top-k verses per chunk
- Merge + dedupe passages with a hard size limit
- Call Ollama once (retry once)
- If Ollama fails, returns context-only answer (never crashes)

## Ollama (free, optional but recommended)
If you install Ollama, VYASA-1 will use it automatically.
1. Install Ollama for Windows
2. Run:
```
ollama pull llama3
```
If Ollama is not running, VYASA-1 still returns a context-only answer.

## Performance note (recommended Ollama model)
For faster responses with lower memory use, use a quantized instruct model like:
- `llama3:8b-instruct-q4_K_M`

Reason (simple): quantized weights are smaller, so it runs faster on typical GPUs/CPUs and uses less RAM/VRAM.

## Confidence + follow-ups (v5 behavior)
Responses now include:
- `confidence`: a simple score based on retrieval strength + grounding checks

At the end of answers, VYASA-1 may add a short “note” when sources emphasize different angles (duty vs detachment, etc.).

## UI formatting note (important)
The frontend is optimized for markdown section output. Best answers follow this structure:
- `Short Answer:`
- `Key Insight:`
- `### Your question`
- `### Scriptural insight`
- `### Simple explanation`
- `### What you can do (if applicable)`
- `### Comparison with modern science (if applicable)`
- `### Final takeaway`

Scripture quotes render best as markdown blockquotes:
`> quoted line here`

Exact verse requests are strict:
- If user asks for `specific line`, `which verse`, `exact reference`, VYASA-1 returns an exact indexed verse when metadata exists (`canto/chapter/verse`).
- If exact verse metadata is missing, VYASA-1 returns: `No exact verse found in indexed corpus`.

---

## HOW IT WORKS

1. **You type a problem** → VYASA-1 detects your emotion (anxiety, grief, anger, etc.)
2. **RAG Search** → It searches thousands of real Vedic verses from the corpus
3. **AI Response** → It generates a compassionate, scripture-backed answer
4. **3 Citations minimum** → Every answer cites real sources with Sanskrit + English

---

## COMMON ERRORS AND FIXES

**Error: `ModuleNotFoundError`**
→ Make sure your virtual environment is activated: `vyasa_env\Scripts\activate`

**Error: `No such file: vyasa_corpus.jsonl`**
→ Run: `python scripts/build_corpus.py`

**Error: `Collection vyasa_corpus not found`**
→ Run: `python scripts/build_index.py`

**Server starts but browser shows nothing**
→ Go to http://127.0.0.1:8000 instead of localhost

**CUDA out of memory during fine-tuning**
→ Open phase3_finetune.py and change `per_device_train_batch_size=2` to `1`

---

## HARDWARE USED
- Intel i9-14900
- 32 GB RAM
- NVIDIA RTX 2000 Ada (16 GB VRAM)
- CUDA 12.x

---

## DATA SOURCES (100% Free & Open)
- bhavykhatri/DharmicData (HuggingFace)
- rahular/itihasa — 93,000+ Sanskrit-English shlokas
- atmabodha/Vedanta_Datasets
- Abhaykoul/Ancient-Indian-Wisdom
- SoumilB7/Indic-Data
- Multiple Gita and Upanishad datasets

---

*ॐ तत् सत् — Om Tat Sat*
