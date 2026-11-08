"""Part D, tasks 1-4: chunk the five policy documents, TF-IDF retrieval, grounded answers via
mock_llm (POLICY ASSISTANT) and an evaluation with overlap 10 and overlap 0.
The refund agent (part_d_agent.py) reuses retrieve() from this module.

Run:  python src/part_d_rag.py
"""
import glob
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import POLICIES, RAW, header
import mock_llm

CHUNK_WORDS = 40
OVERLAP = 10

SYSTEM_PROMPT = ("You are the Instamart POLICY ASSISTANT. Answer the question using ONLY the policy CONTEXT "
                 "supplied in the user message. Do not use outside knowledge. If the context does not contain "
                 "the answer, reply exactly: I could not find that information in the policy documents.")


# ---------------------------------------------------------------- task 1: chunking
def chunk_words(words, size=CHUNK_WORDS, overlap=OVERLAP):
    """Windows of `size` words that start every (size - overlap) words; stop after the first
    window whose end reaches or passes the last word."""
    step = size - overlap
    chunks, start = [], 0
    while True:
        chunks.append(words[start:start + size])
        if start + size >= len(words):
            break
        start += step
    return chunks


def load_chunks(size=CHUNK_WORDS, overlap=OVERLAP):
    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICIES, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        for i, w in enumerate(chunk_words(words, size, overlap)):
            chunks.append({"source": os.path.basename(path), "chunk_no": i, "text": " ".join(w)})
    return chunks


# ---------------------------------------------------------------- task 2: vectorise and retrieve
class PolicyIndex:
    def __init__(self, size=CHUNK_WORDS, overlap=OVERLAP):
        self.chunks = load_chunks(size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        q = self.vectorizer.transform([query])                 # same fitted vectorizer
        scores = cosine_similarity(q, self.matrix)[0]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(scores[i]), 4),
                 "text": self.chunks[i]["text"]} for i in order]

    # ------------------------------------------------------------ task 3: augment and answer
    def answer(self, question, k=3):
        hits = self.retrieve(question, k)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        resp = mock_llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": user}])
        return {"answer": resp.choices[0].message.content, "sources": [h["source"] for h in hits],
                "tokens": resp.usage.total_tokens}


_DEFAULT = None


def default_index():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PolicyIndex()
    return _DEFAULT


def retrieve(query, k=3):
    return default_index().retrieve(query, k)


def answer(question):
    return default_index().answer(question)


# ---------------------------------------------------------------- task 4: evaluation
def evaluate(index, label):
    qa = pd.read_csv(f"{RAW}/policy_eval.csv", keep_default_na=False)
    rows = []
    for q in qa.itertuples(index=False):
        res = index.answer(q.question)
        answerable = q.expected_doc != "NONE"
        rows.append({"question": q.question, "expected_doc": q.expected_doc,
                     "doc_in_top3": (q.expected_doc in res["sources"]) if answerable else None,
                     "fact_in_answer": (q.expected_fact in res["answer"]) if answerable else None,
                     "fallback": res["answer"].startswith("I could not find"),
                     "sources": ",".join(s.replace("_policy.txt", "") for s in res["sources"]),
                     "answer": res["answer"]})
    df = pd.DataFrame(rows)
    ans, una = df[df["expected_doc"] != "NONE"], df[df["expected_doc"] == "NONE"]
    summary = {"chunks": len(index.chunks),
               "expected_doc_in_top3": f"{int(ans['doc_in_top3'].sum())}/{len(ans)}",
               "expected_fact_in_answer": f"{int(ans['fact_in_answer'].sum())}/{len(ans)}",
               "unanswerable_fallback": f"{int(una['fallback'].sum())}/{len(una)}"}
    print(f"\n[{label}] {summary}")
    with pd.option_context("display.max_colwidth", 90, "display.width", 250):
        print(df[["question", "doc_in_top3", "fact_in_answer", "fallback", "sources", "answer"]].to_string(index=False))
    return df, summary


def main():
    header("PART D - TASK 1: CHUNKS (40 words, overlap 10)")
    idx = default_index()
    counts = pd.Series([c["source"] for c in idx.chunks]).value_counts()
    print(f"total chunks: {len(idx.chunks)}")
    print(counts.to_string())
    for path in sorted(glob.glob(os.path.join(POLICIES, "*.txt"))):
        print(f"  {os.path.basename(path)}: {len(open(path, encoding='utf-8').read().split())} words")

    header("PART D - TASK 2: TF-IDF RETRIEVAL")
    print(f"vocabulary size: {len(idx.vectorizer.vocabulary_)}")
    for h in retrieve("refund policy late delivery"):
        print(f"  {h['source']:<22} score {h['score']:.4f}  {h['text'][:70]}...")

    header("PART D - TASK 3: GROUNDED ANSWER EXAMPLE")
    hits = retrieve("Who must approve a refund above INR 1000?")
    print("user message sent to the model:\nCONTEXT:\n" + "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
          + "\nQUESTION:\nWho must approve a refund above INR 1000?")
    print("\nanswer():", answer("Who must approve a refund above INR 1000?"))

    header("PART D - TASK 4: EVALUATION ON policy_eval.csv")
    evaluate(idx, "overlap 10")
    df0, _ = evaluate(PolicyIndex(CHUNK_WORDS, 0), "overlap 0")
    broken = df0[(df0["expected_doc"] != "NONE") & (~df0["doc_in_top3"].astype(bool) | ~df0["fact_in_answer"].astype(bool))]
    print("\nquestions that break with overlap 0:", broken["question"].tolist())
    for c in PolicyIndex(CHUNK_WORDS, 0).chunks:
        if c["source"] in set(broken["expected_doc"]):
            print(f"  [{c['source']} chunk {c['chunk_no']}] ...{c['text'][-90:]}")


if __name__ == "__main__":
    main()
