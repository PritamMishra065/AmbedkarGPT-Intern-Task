# evaluation.py (final)
"""
Evaluation script (final)
- Reads corpus/*.txt and test_dataset.json from repo root
- Builds Chroma DBs for chunk sizes and runs retrieval+generation
- Saves results to results/test_results.json
- Uses a SentenceTransformer wrapper (STEmbeddings) compatible with Chroma
Notes:
- Corpus must be in ./corpus/*.txt (speech1.txt ... speech6.txt)
- test_dataset.json must be a flat list of question objects in repo root
- Uploaded file path (not used directly): /mnt/data/AI Intern_Assignment Test_KalpIT@1 (1).pdf
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import re

# ---- Config ----
CORPUS_DIR = "corpus"
CHUNK_CONFIGS = {
    "small": {"chunk_size": 250, "chunk_overlap": 50},
    "medium": {"chunk_size": 550, "chunk_overlap": 100},
    "large": {"chunk_size": 900, "chunk_overlap": 150},
}
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# Local uploaded PDF path (kept for traceability; not used by default)
UPLOADED_PDF_PATH = "/mnt/data/AI Intern_Assignment Test_KalpIT@1 (1).pdf"

# ---- Libraries ----
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from rouge_score.rouge_scorer import RougeScorer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu  # we'll pass token lists we generate

# ---- Utilities ----
_token_re = re.compile(r"\w+")
def simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _token_re.findall(text.lower())

# ---- Embedding wrapper for Chroma ----
class STEmbeddings:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        print(f"[+] Loading embeddings model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        v = self.model.encode([text], show_progress_bar=False)[0]
        return v.tolist()

# ---- Corpus loader ----
def load_corpus_texts(corpus_dir: str) -> Dict[str, str]:
    p = Path(corpus_dir)
    if not p.exists() or not any(p.glob("*.txt")):
        raise FileNotFoundError(f"No text files found in corpus directory: {corpus_dir}")
    files = sorted(p.glob("*.txt"))
    out = {}
    for f in files:
        out[f.name] = f.read_text(encoding="utf-8")
    return out

# ---- Build Chroma safely ----
def build_chroma_for_config(docs: Dict[str, str], config_name: str, persist_base: str, embed_model):
    cfg = CHUNK_CONFIGS[config_name]
    persist_dir = f"{persist_base}_{config_name}"
    os.makedirs(persist_dir, exist_ok=True)

    # if DB exists and non-empty, reuse
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            return Chroma(persist_directory=persist_dir, embedding_function=embed_model)
        except TypeError:
            return Chroma(persist_directory=persist_dir, embedding=embed_model)

    texts, metadatas = [], []
    splitter = CharacterTextSplitter(separator="\n", chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
    for doc_name, text in docs.items():
        if not text or not text.strip():
            continue
        chunks = splitter.split_text(text)
        for i, c in enumerate(chunks):
            if c and c.strip():
                texts.append(c)
                metadatas.append({"source": doc_name, "chunk_index": i})

    if len(texts) == 0:
        print(f"[!] No chunks generated for config '{config_name}' — skipping Chroma build.")
        return None

    try:
        vect = Chroma.from_texts(texts=texts, embedding=embed_model, metadatas=metadatas, persist_directory=persist_dir)
    except TypeError:
        vect = Chroma.from_texts(texts=texts, embedding_function=embed_model, metadatas=metadatas, persist_directory=persist_dir)

    try:
        vect.persist()
    except Exception:
        pass
    return vect

# ---- Retriever helper ----
def retrieve_topk(retriever, q: str, k: int):
    for name in ("get_relevant_documents", "_get_relevant_documents", "retrieve", "get_relevant"):
        if hasattr(retriever, name):
            method = getattr(retriever, name)
            try:
                return method(q)
            except TypeError:
                try:
                    return method(q, run_manager=None)
                except Exception:
                    try:
                        return method(query=q)
                    except Exception:
                        continue
    if hasattr(retriever, "similarity_search"):
        try:
            return retriever.similarity_search(q, k=k)
        except Exception:
            return []
    return []

# ---- Metrics ----
def compute_metrics_for_question(retrieved_docs, ground_truth_docs: List[str], k: int):
    sources = []
    for d in retrieved_docs:
        s = getattr(d, "metadata", None)
        if s and "source" in s:
            sources.append(s["source"])
        else:
            try:
                sources.append(d.metadata["source"])
            except Exception:
                sources.append(None)
    hit = int(any(s in ground_truth_docs for s in sources if s))
    rr = 0.0
    for idx, s in enumerate(sources, start=1):
        if s and s in ground_truth_docs:
            rr = 1.0 / idx
            break
    prec = sum(1 for s in sources[:k] if s and s in ground_truth_docs) / max(1, k)
    return {"hit": hit, "mrr": rr, "prec_at_k": prec, "retrieved_sources": sources}

def score_texts(gen: str, gold: str, embed_model: STEmbeddings):
    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    try:
        rouge = scorer.score(gold or "", gen or "")["rougeL"].fmeasure
    except Exception:
        rouge = 0.0
    try:
        ref_tokens = [simple_tokenize(gold)]
        hyp = simple_tokenize(gen)
        bleu = sentence_bleu(ref_tokens, hyp, weights=(0.5, 0.5))
    except Exception:
        bleu = 0.0
    try:
        # use the underlying SentenceTransformer model for embeddings
        v = embed_model.model.encode([gen or "", gold or ""], show_progress_bar=False)
        cos = float(cosine_similarity([v[0]], [v[1]])[0][0])
    except Exception:
        cos = 0.0
    gen_unigrams = set(simple_tokenize(gen))
    gold_unigrams = set(simple_tokenize(gold))
    faith = (len(gen_unigrams & gold_unigrams) / len(gen_unigrams)) if gen_unigrams else 0.0
    return {"rouge_l": rouge, "bleu": bleu, "cosine": cos, "faith_overlap": faith}

# ---- LLM wrapper ----
def call_llm_and_extract(llm, prompt: str) -> str:
    try:
        r = llm(prompt)
    except TypeError:
        try:
            r = llm.generate([prompt])
        except Exception as e:
            return f"LLM error: {e}"
    if isinstance(r, str):
        return r
    if hasattr(r, "text"):
        return r.text
    if hasattr(r, "generations"):
        try:
            return r.generations[0][0].text
        except Exception:
            pass
    return str(r)

# ---- Main runner ----
def run_evaluation(out_dir: str, k: int):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_corpus_texts(CORPUS_DIR)

    td_path = Path("test_dataset.json")
    if not td_path.exists():
        print("test_dataset.json not found in repo root. Please create it.")
        return

    with td_path.open("r", encoding="utf-8") as fh:
        test_dataset = json.load(fh)

    embed_model = STEmbeddings(EMBED_MODEL_NAME)

    results: Dict[str, Any] = {"per_chunk_config": {}}
    for cfg_name in CHUNK_CONFIGS:
        print("Running config:", cfg_name)
        vect = build_chroma_for_config(docs, cfg_name, persist_base="chroma_db", embed_model=embed_model)

        if vect is None:
            print(f"[!] Skipping config '{cfg_name}' because no chunks were generated.")
            results["per_chunk_config"][cfg_name] = {"per_question": [], "aggregate": {"hit_avg": 0.0, "mrr_avg": 0.0, "rouge_avg": 0.0}}
            continue

        try:
            retriever = vect.as_retriever(search_kwargs={"k": k})
        except Exception:
            retriever = vect

        results["per_chunk_config"][cfg_name] = {"per_question": []}
        llm = Ollama(model="mistral", temperature=0.0)

        for item in tqdm(test_dataset, desc=f"Questions ({cfg_name})"):
            qid = item.get("id") or item.get("question")[:30]
            q = item["question"]
            gold = item.get("ground_truth", "")
            sources_gt = item.get("source_documents", [])

            retrieved = retrieve_topk(retriever, q, k=k) or []
            retr_metrics = compute_metrics_for_question(retrieved, sources_gt, k)

            ctx = []
            for d in retrieved:
                txt = getattr(d, "page_content", None) if not isinstance(d, str) else d
                if txt is None:
                    try:
                        txt = d["text"]
                    except Exception:
                        txt = ""
                src = getattr(d, "metadata", {}).get("source", "")
                ctx.append(f"Chunk (source={src}):\n{txt}")

            context = "\n\n---\n\n".join(ctx)[:3000]
            prompt = f"Use ONLY the CONTEXT to answer. If not present, say 'I don't know'.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{q}\n\nANSWER:"
            gen = call_llm_and_extract(llm, prompt=prompt)

            scores = score_texts(gen, gold, embed_model)

            rec = {
                "id": qid,
                "question": q,
                "ground_truth": gold,
                "generated": gen,
                "retrieval": retr_metrics,
                "scores": scores
            }
            results["per_chunk_config"][cfg_name]["per_question"].append(rec)

        perq = results["per_chunk_config"][cfg_name]["per_question"]
        hit_avg = sum(p["retrieval"]["hit"] for p in perq) / max(1, len(perq))
        mrr_avg = sum(p["retrieval"]["mrr"] for p in perq) / max(1, len(perq))
        rouge_avg = sum(p["scores"]["rouge_l"] for p in perq) / max(1, len(perq))
        results["per_chunk_config"][cfg_name]["aggregate"] = {"hit_avg": hit_avg, "mrr_avg": mrr_avg, "rouge_avg": rouge_avg}

    (out_dir / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved results to", out_dir / "test_results.json")

# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--k", type=int, default=5, help="top-k retrieval")
    args = parser.parse_args()
    run_evaluation(args.out, k=args.k)
