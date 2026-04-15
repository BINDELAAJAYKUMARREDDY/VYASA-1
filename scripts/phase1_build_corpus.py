import argparse
import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

import requests
from datasets import load_dataset


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DOWNLOADS_DIR = DATA_DIR / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

CORPUS_FILE = DATA_DIR / "vyasa_corpus.jsonl"
QA_FILE = DATA_DIR / "vyasa_qa.jsonl"


ALLOWED_SOURCES = [
    "bhavykhatri/DharmicData",
    "atmabodha/Vedanta_Datasets",
    "rahular/itihasa",
    "SoumilB7/Indic-Data",
    "Abhaykoul/Ancient-Indian-Wisdom",
    "GRETIL",
]


def _now():
    return time.strftime("%H:%M:%S")


def _print(msg):
    print(f"[{_now()}] {msg}")


def _safe_text(x):
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _clean_space(s):
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _guess_lang_fields(row):
    sans = ""
    eng = ""

    for k in ["sanskrit", "shloka", "sloka", "devanagari", "text_sa", "sa"]:
        if k in row and row.get(k):
            sans = _safe_text(row.get(k))
            break

    for k in ["english", "translation", "meaning", "text_en", "en", "text"]:
        if k in row and row.get(k):
            eng = _safe_text(row.get(k))
            break

    return _clean_space(sans), _clean_space(eng)


def _keywords_from_text(text):
    t = (text or "").lower()
    keys = []

    groups = {
        "anxiety": ["anxiet", "worry", "fear", "panic", "nervous", "stress"],
        "grief": ["grief", "mourn", "loss", "died", "death", "sorrow"],
        "anger": ["anger", "furious", "rage", "hate"],
        "dharma": ["dharma", "duty", "right", "wrong", "ethic", "virtue"],
        "karma": ["karma", "action", "deed", "fruit", "result"],
        "bhakti": ["bhakti", "devotion", "worship", "lord", "krishna", "rama", "shiva", "devi"],
        "moksha": ["moksha", "mukti", "liberation", "freedom"],
        "mind": ["mind", "thought", "control", "calm", "peace"],
        "cosmology": ["universe", "cosmos", "creation", "time cycle", "yuga", "loka"],
        "astronomy": ["sun", "moon", "star", "planet", "eclipse", "sky"],
    }

    for k, triggers in groups.items():
        for w in triggers:
            if w in t:
                keys.append(k)
                break

    if not keys:
        return ["general"]
    return sorted(set(keys))


def _category_from_keywords(keys):
    if any(k in keys for k in ["astronomy", "cosmology"]):
        return "scientific"
    if any(k in keys for k in ["anxiety", "grief", "anger", "mind"]):
        return "emotional"
    if any(k in keys for k in ["dharma", "karma"]):
        return "moral"
    return "general"


def _write_jsonl(path, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _download_file(url, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    _print(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest_path


def _extract_zip(zip_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted.ok"
    if marker.exists():
        return out_dir

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    marker.write_text("ok", encoding="utf-8")
    return out_dir


def load_hf_dataset(name, split=None):
    if split:
        return load_dataset(name, split=split, trust_remote_code=True)
    ds = load_dataset(name, trust_remote_code=True)
    if "train" in ds:
        return ds["train"]
    first = list(ds.keys())[0]
    return ds[first]


def build_corpus(limit_each=0):
    _print("VYASA-1 Phase 1: Corpus Builder Starting (strict real sources)")
    _print("Allowed scripture sources: " + ", ".join(ALLOWED_SOURCES))

    rows = []
    seen = set()

    def add_row(source, sanskrit, english):
        s = _clean_space(sanskrit)
        e = _clean_space(english)
        if not e:
            return

        key = (source, s[:200], e[:400])
        if key in seen:
            return
        seen.add(key)

        keys = _keywords_from_text(e + " " + s)
        cat = _category_from_keywords(keys)

        rid = f"vyasa_{len(rows)}"
        rows.append(
            {
                "id": rid,
                "source": _clean_space(source)[:300],
                "sanskrit": s,
                "english": e,
                "keywords": keys,
                "category": cat,
            }
        )

    def load_csv_file(path_obj, source_prefix):
        import csv

        try:
            with open(path_obj, "r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if limit_each and i >= limit_each:
                        break
                    if not isinstance(row, dict):
                        continue
                    src = _safe_text(row.get("source") or row.get("book") or row.get("ref") or source_prefix)
                    sans = _safe_text(
                        row.get("sanskrit")
                        or row.get("shloka")
                        or row.get("sloka")
                        or row.get("devanagari")
                        or row.get("verse_sanskrit")
                        or row.get("sn")
                        or ""
                    )
                    eng = _safe_text(
                        row.get("english")
                        or row.get("translation")
                        or row.get("meaning")
                        or row.get("verse_english")
                        or row.get("en")
                        or row.get("text")
                        or ""
                    )

                    if not src or src == source_prefix:
                        chap = row.get("chapter") or row.get("canto") or row.get("book") or ""
                        ver = row.get("verse") or row.get("verse_number") or row.get("id") or ""
                        if chap and ver:
                            src = f"{source_prefix} {chap}.{ver}"
                        else:
                            src = source_prefix

                    add_row(src, sans, eng)
        except Exception:
            return

    # 1) bhavykhatri/DharmicData (GitHub)
    _print("Downloading: bhavykhatri/DharmicData (GitHub)")
    try:
        url = "https://github.com/bhavykhatri/DharmicData/archive/refs/heads/main.zip"
        zip_path = _download_file(url, DOWNLOADS_DIR / "DharmicData-main.zip")
        out_dir = _extract_zip(zip_path, DOWNLOADS_DIR / "DharmicData-main")
        base = out_dir / "DharmicData-main"

        # Support JSON / JSONL / CSV (repo may change structure over time)
        json_files = list(base.rglob("*.json")) + list(base.rglob("*.jsonl"))
        csv_files = list(base.rglob("*.csv"))

        for p in json_files:
            if limit_each and len(rows) > limit_each * 5:
                break
            if p.suffix.lower() == ".jsonl":
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f):
                            if limit_each and i >= limit_each:
                                break
                            try:
                                row = json.loads(line)
                            except Exception:
                                continue
                            if not isinstance(row, dict):
                                continue
                            src = _safe_text(row.get("source") or row.get("book") or p.name)
                            s, e = _guess_lang_fields(row)
                            add_row(src, s, e)
                except Exception:
                    pass
            else:
                try:
                    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    continue
                items = data if isinstance(data, list) else [data]
                for i, row in enumerate(items):
                    if limit_each and i >= limit_each:
                        break
                    if not isinstance(row, dict):
                        continue
                    src = _safe_text(row.get("source") or row.get("book") or p.name)
                    s, e = _guess_lang_fields(row)
                    add_row(src, s, e)

        for p in csv_files:
            load_csv_file(p, "DharmicData")

        _print(f"Loaded DharmicData. rows: {len(rows)} total so far")
    except Exception as ex:
        _print(f"FAILED DharmicData: {ex}")

    # 2) atmabodha/Vedanta_Datasets (GitHub)
    #    NOTE: This repo is GitHub-only per your requirements.
    _print("Downloading: atmabodha/Vedanta_Datasets (GitHub)")
    try:
        url = "https://github.com/atmabodha/Vedanta_Datasets/archive/refs/heads/main.zip"
        zip_path = _download_file(url, DOWNLOADS_DIR / "Vedanta_Datasets-main.zip")
        out_dir = _extract_zip(zip_path, DOWNLOADS_DIR / "Vedanta_Datasets-main")
        base = out_dir / "Vedanta_Datasets-main"
        for p in base.rglob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            items = data if isinstance(data, list) else [data]
            for i, row in enumerate(items):
                if limit_each and i >= limit_each:
                    break
                if not isinstance(row, dict):
                    continue
                src = _safe_text(row.get("source") or row.get("book") or p.name)
                s, e = _guess_lang_fields(row)
                add_row(src, s, e)

        for p in base.rglob("*.csv"):
            load_csv_file(p, "Vedanta_Datasets")

        _print(f"Loaded Vedanta_Datasets. rows: {len(rows)} total so far")
    except Exception as ex:
        _print(f"FAILED Vedanta_Datasets: {ex}")

    # 3) rahular/itihasa (GitHub)
    _print("Downloading: rahular/itihasa (GitHub)")
    try:
        url = "https://github.com/rahular/itihasa/archive/refs/heads/master.zip"
        zip_path = _download_file(url, DOWNLOADS_DIR / "itihasa-master.zip")
        out_dir = _extract_zip(zip_path, DOWNLOADS_DIR / "itihasa-master")
        base = out_dir / "itihasa-main"

        # Typical itihasa format: parallel files train.sn/train.en etc.
        data_dir = base / "data"
        pairs = [("train.sn", "train.en"), ("dev.sn", "dev.en"), ("test.sn", "test.en")]
        loaded = 0

        for sn_name, en_name in pairs:
            sn_path = data_dir / sn_name
            en_path = data_dir / en_name
            if not (sn_path.exists() and en_path.exists()):
                continue

            with open(sn_path, "r", encoding="utf-8", errors="ignore") as fsn, open(
                en_path, "r", encoding="utf-8", errors="ignore"
            ) as fen:
                for i, (sn_line, en_line) in enumerate(zip(fsn, fen)):
                    if limit_each and i >= limit_each:
                        break
                    s = _clean_space(sn_line)
                    e = _clean_space(en_line)
                    if not e:
                        continue
                    add_row("Itihasa", s, e)
                    loaded += 1

        # Fallback if repo layout changes
        for p in base.rglob("*.csv"):
            load_csv_file(p, "Itihasa")
        for p in base.rglob("*.jsonl"):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f):
                        if limit_each and i >= limit_each:
                            break
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(row, dict):
                            continue
                        src = _safe_text(row.get("source") or row.get("book") or "Itihasa")
                        s = _safe_text(row.get("shloka") or row.get("sanskrit") or "")
                        e = _safe_text(row.get("translation") or row.get("english") or "")
                        add_row(src, s, e)
            except Exception:
                pass

        _print(f"Loaded itihasa. rows: {len(rows)} total so far")
    except Exception as ex:
        _print(f"FAILED itihasa: {ex}")

    # 4) Supplementary HF datasets (optional, but on your allowed list)
    for name in ["SoumilB7/Indic-Data", "Abhaykoul/Ancient-Indian-Wisdom"]:
        _print(f"Loading: {name} (HF)")
        try:
            ds = load_hf_dataset(name)
            for i, row in enumerate(ds):
                if limit_each and i >= limit_each:
                    break
                if not isinstance(row, dict):
                    continue
                src = _safe_text(row.get("source") or row.get("book") or name)
                s, e = _guess_lang_fields(row)
                add_row(src, s, e)
            _print(f"Loaded {name}. rows: {len(rows)} total so far")
        except Exception as ex:
            _print(f"FAILED {name}: {ex}")

    # 5) GRETIL zip (direct download)
    #    You requested: https://gretil.sub.uni-goettingen.de/gretil/1_sanskr.zip
    _print("Downloading: GRETIL full corpus zip (direct)")
    try:
        gretil_url = "https://gretil.sub.uni-goettingen.de/gretil/1_sanskr.zip"
        zip_path = _download_file(gretil_url, DOWNLOADS_DIR / "gretil_1_sanskr.zip")
        out_dir = _extract_zip(zip_path, DOWNLOADS_DIR / "gretil_1_sanskr")

        # GRETIL is big and formats vary. For Phase 1 we do a safe, simple parse:
        # - take short text chunks (paragraph-like) as "english" ONLY if it contains enough Latin letters
        # - otherwise store in "sanskrit" and leave english empty (english required for indexing now)
        # This keeps the pipeline reliable, and you can later add a better Sanskrit transliteration/translation stage.
        base = out_dir
        txt_files = list(base.rglob("*.txt"))
        picked = 0
        for p in txt_files:
            if limit_each and picked >= max(2000, limit_each):
                break
            try:
                raw = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            raw = raw.replace("\r\n", "\n")
            parts = [x.strip() for x in re.split(r"\n{2,}", raw) if x.strip()]
            for part in parts[:20]:
                if limit_each and picked >= max(2000, limit_each):
                    break

                latin = sum(1 for c in part if "a" <= c.lower() <= "z")
                if latin < 60:
                    continue

                src = f"GRETIL:{p.relative_to(base).as_posix()}"
                add_row(src, "", part)
                picked += 1

        _print(f"Loaded GRETIL (english-like chunks). rows: {len(rows)} total so far")
    except Exception as ex:
        _print(f"FAILED GRETIL: {ex}")

    return rows


def _score_overlap(a, b):
    a = re.findall(r"[a-z]{3,}", (a or "").lower())
    b = re.findall(r"[a-z]{3,}", (b or "").lower())
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / (len(sa) + 1e-9)


def _pick_refs(corpus, main_entry, k=3):
    target = (main_entry.get("english", "") + " " + main_entry.get("sanskrit", "")).strip()
    scored = []
    for e in corpus:
        if e["id"] == main_entry["id"]:
            continue
        s = _score_overlap(target, e.get("english", ""))
        if s > 0:
            scored.append((s, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:k]]


def generate_qa_from_real_corpus(corpus_rows, qa_count=300, seed=42):
    import random

    random.seed(seed)
    if not corpus_rows:
        return []

    usable = [e for e in corpus_rows if (e.get("english") or "").strip()]
    if not usable:
        return []

    qa = []
    for i in range(min(qa_count, len(usable))):
        main = random.choice(usable)
        refs = _pick_refs(usable, main, k=3)

        topic = (main.get("keywords") or ["life"])[0]
        cat = main.get("category", "general")

        prompt = f"I am facing a real-life problem related to {topic}. Please give a dharmic, practical solution with exact citations."

        lines = []
        lines.append("I hear you. Let us handle this with steadiness and compassion.")
        lines.append("")
        lines.append("Here are scripture passages from the real corpus (Sanskrit + English when available):")
        lines.append("")

        def add_cite(entry, idx):
            lines.append(f"Verse {idx} — {entry.get('source','')}".strip())
            s = (entry.get("sanskrit") or "").strip()
            e = (entry.get("english") or "").strip()
            if s:
                lines.append("Sanskrit: " + s[:800])
            if e:
                lines.append("English: " + e[:900])
            lines.append("")

        add_cite(main, 1)
        for j, r in enumerate(refs[:2], start=2):
            add_cite(r, j)

        lines.append("What this means (beginner-friendly):")
        lines.append("- Keep your mind steady; do the next right action without panic.")
        lines.append("- Choose truth and non-harm; avoid actions that create bigger suffering later.")
        lines.append("")
        lines.append("Action steps for today:")
        lines.append("1. Write the next smallest action you can do in 10 minutes, and do it now.")
        lines.append("2. Do 5 minutes of slow breathing (inhale 4s, exhale 6s) to calm the mind.")
        lines.append("3. Read one cited verse again slowly and note 1 practical meaning for your life.")

        qa.append(
            {
                "id": f"qa_{i}",
                "instruction": prompt,
                "response": "\n".join(lines).strip(),
                "category": cat,
                "sources_used": [main.get("source", "")] + [r.get("source", "") for r in refs[:2]],
            }
        )

    return qa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-each", type=int, default=0, help="limit rows per dataset (0 = no limit)")
    parser.add_argument("--qa-count", type=int, default=300, help="number of QA examples to generate")
    args = parser.parse_args()

    corpus = build_corpus(limit_each=args.limit_each)
    _print(f"Writing corpus -> {CORPUS_FILE}")
    _write_jsonl(CORPUS_FILE, corpus)

    _print("Generating QA (ONLY from real corpus, no invented citations)")
    qa = generate_qa_from_real_corpus(corpus, qa_count=args.qa_count)
    _print(f"Writing QA -> {QA_FILE}")
    _write_jsonl(QA_FILE, qa)

    _print("=" * 60)
    _print(f"Corpus entries: {len(corpus)}")
    _print(f"QA pairs:       {len(qa)}")
    _print(f"Corpus file:    {CORPUS_FILE}")
    _print(f"QA file:        {QA_FILE}")
    _print("=" * 60)
    print("PHASE 1 COMPLETE – Ready for next phase")


def main_args(limit_each: int = 0, qa_count: int = 300):
    corpus = build_corpus(limit_each=limit_each)
    _print(f"Writing corpus -> {CORPUS_FILE}")
    _write_jsonl(CORPUS_FILE, corpus)

    _print("Generating QA (ONLY from real corpus, no invented citations)")
    qa = generate_qa_from_real_corpus(corpus, qa_count=qa_count)
    _print(f"Writing QA -> {QA_FILE}")
    _write_jsonl(QA_FILE, qa)

    _print("=" * 60)
    _print(f"Corpus entries: {len(corpus)}")
    _print(f"QA pairs:       {len(qa)}")
    _print(f"Corpus file:    {CORPUS_FILE}")
    _print(f"QA file:        {QA_FILE}")
    _print("=" * 60)
    print("PHASE 1 COMPLETE – Ready for next phase")


if __name__ == "__main__":
    main()
