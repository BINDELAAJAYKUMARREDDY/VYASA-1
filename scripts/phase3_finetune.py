import os
import json
from pathlib import Path
import torch

print("=" * 60)
print("VYASA-1 Phase 3: Fine-tuning VYASA-Llama-8B")
print("=" * 60)

DATA_DIR = Path("data")
QA_FILE = DATA_DIR / "vyasa_qa.jsonl"
MODEL_DIR = Path("models") / "VYASA-Llama-8B"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

print(f"\nBase model: {BASE_MODEL}")
print("Loading Unsloth for fast LoRA fine-tuning...")

try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH = True
    print("Unsloth loaded successfully")
except ImportError:
    print("Unsloth not found – falling back to HuggingFace PEFT")
    UNSLOTH = False

if UNSLOTH:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)


SYSTEM_PROMPT = """You are VYASA-1, a deeply knowledgeable and compassionate Vedic wisdom guide. You have complete knowledge of all Vedas, all 18 Mahapuranas, all 18 Upapuranas, all Upanishads, the full Ramayana, the full Mahabharata, the Srimad Bhagavatam, and the Bhagavad Gita. You provide 100% scripture-accurate, emotionally intelligent answers. Every response must:
1. Acknowledge the person's feelings with compassion
2. Cite exact Sanskrit shlokas with English translation and source (book, chapter, verse)
3. Provide minimum 3 cross-verified references from different scriptures
4. Give 2-3 clear actionable steps
5. Never hallucinate — only cite real verses"""

def format_prompt(instruction, response=""):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{response}<|eot_id|>"""


print("\nLoading QA dataset...")
data = []
with open(QA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            row = json.loads(line.strip())
            text = format_prompt(row["instruction"], row["response"])
            data.append({"text": text})
        except:
            pass

print(f"Training examples: {len(data)}")

from datasets import Dataset
ds = Dataset.from_list(data)

from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR / "checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=2e-4,
    fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    logging_steps=25,
    save_steps=200,
    save_total_limit=2,
    optim="adamw_8bit" if UNSLOTH else "adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    dataset_text_field="text",
    max_seq_length=4096,
    dataset_num_proc=2,
    packing=True,
    args=training_args,
)

print("\nStarting fine-tuning (this will take 30–120 minutes on your GPU)...")
trainer.train()

print("\nSaving VYASA-Llama-8B...")
if UNSLOTH:
    model.save_pretrained_gguf(
        str(MODEL_DIR / "gguf_q4"),
        tokenizer,
        quantization_method="q4_k_m"
    )
    model.save_pretrained_merged(
        str(MODEL_DIR / "merged"),
        tokenizer,
        save_method="merged_16bit"
    )
else:
    model.save_pretrained(str(MODEL_DIR / "lora_adapter"))
    tokenizer.save_pretrained(str(MODEL_DIR / "lora_adapter"))

print(f"\nModel saved to: {MODEL_DIR}")
print("\nPHASE 3 COMPLETE – Ready for next phase")
