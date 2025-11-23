# main.py (trimmed)
import os
from typing import List, Any
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer

# try the uploaded PDF first, else speech.txt
speech_path = r"/mnt/data/AI Intern_Assignment Test_KalpIT@1 (1).pdf"
if not os.path.exists(speech_path):
    speech_path = "speech.txt"

class E:
    def __init__(self, m="all-MiniLM-L6-v2"):
        self.m = SentenceTransformer(m)
    def embed_documents(self, texts: List[str]):
        vecs = self.m.encode(texts, show_progress_bar=False)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]
    def embed_query(self, text: str):
        v = self.m.encode([text], show_progress_bar=False)[0]
        return v.tolist() if hasattr(v, "tolist") else list(v)

def build_vectorstore(persist="chroma_db"):
    docs = TextLoader(speech_path, encoding="utf-8").load()
    texts = [d.page_content for d in CharacterTextSplitter("\n", 400, 50).split_documents(docs)]
    vec = E()
    try:
        db = Chroma.from_texts(texts=texts, embedding=vec, persist_directory=persist)
    except TypeError:
        db = Chroma.from_texts(texts=texts, embedding_function=vec, persist_directory=persist)
    try:
        db.persist()
    except: pass
    return db

def get_retriever(persist="chroma_db", k=3):
    vec = E()
    try:
        db = Chroma(persist_directory=persist, embedding_function=vec)
    except TypeError:
        db = Chroma(persist_directory=persist, embedding=vec)
    try:
        return db.as_retriever(search_kwargs={"k": k})
    except:
        return db

def build_prompt(docs: List[Any], q: str, max_chars=3000) -> str:
    chunks, total = [], 0
    for d in docs:
        t = getattr(d, "page_content", d) if not isinstance(d, str) else d
        if not t: continue
        if total + len(t) > max_chars:
            remaining = max_chars - total
            if remaining <= 0: break
            chunks.append(t[:remaining]); break
        chunks.append(t); total += len(t)
    ctx = "\n\n---\n\n".join(chunks)
    return f"Use ONLY the CONTEXT to answer.\nCONTEXT:\n{ctx}\nQUESTION:\n{q}"

def call_llm(llm, prompt: str) -> str:
    r = None
    try:
        r = llm(prompt)
    except TypeError:
        try: r = llm.generate([prompt])
        except Exception as e: return str(e)
    if isinstance(r, str): return r
    if hasattr(r, "text"): return r.text
    if hasattr(r, "generations"):
        try: return r.generations[0][0].text
        except: pass
    return str(r)

def retrieve(retriever, q, k=3):
    # try common methods; if method requires run_manager, call with run_manager=None
    names = ("get_relevant_documents", "_get_relevant_documents", "retrieve", "get_relevant")
    for n in names:
        if hasattr(retriever, n):
            m = getattr(retriever, n)
            try:
                return m(q)
            except TypeError:
                try:
                    return m(q, run_manager=None)
                except Exception:
                    try:
                        return m(query=q)
                    except Exception:
                        pass
    if hasattr(retriever, "similarity_search"):
        try: return retriever.similarity_search(q, k=k)
        except: pass
    return []

if __name__ == "__main__":
    if not os.path.exists("chroma_db") or not os.listdir("chroma_db"):
        build_vectorstore()
    retriever = get_retriever()
    llm = Ollama(model="mistral", temperature=0.0)
    while True:
        q = input("Question: ").strip()
        if not q or q.lower() in ("exit", "quit"): break
        docs = retrieve(retriever, q, k=3) or []
        prompt = build_prompt(docs, q)
        print("Ans:",call_llm(llm, prompt), "\n")
