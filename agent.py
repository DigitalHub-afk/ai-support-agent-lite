"""AI Support Agent Lite: answers questions from your documents (RAG, no memory, no tools)."""
import os
import re
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

MODEL = os.getenv("MODEL", "claude-sonnet-4-6")
KB_DIR = Path(os.getenv("KB_DIR", "knowledge"))
BUSINESS_NAME = os.getenv("BUSINESS_NAME", "My Business")

SYSTEM_PROMPT = f"""You are the support assistant for {BUSINESS_NAME}.
Answer only with the information in the provided context.
If the answer is not there, say you don't know.
Reply in the customer's language and cite the source in brackets, e.g. [shipping.md]."""


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    chunks, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


class KnowledgeBase:
    def __init__(self, folder: Path):
        self.items = [
            (p.name, c)
            for p in sorted(folder.glob("*"))
            if p.suffix.lower() in {".md", ".txt"}
            for c in chunk_text(p.read_text(encoding="utf-8"))
        ]
        if not self.items:
            raise SystemExit(f"No .md or .txt documents found in {folder}/")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([t for _, t in self.items])

    def search(self, query: str, k: int = 3) -> list[tuple[str, str]]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        return [self.items[i] for i in scores.argsort()[::-1][:k] if scores[i] > 0]


def answer(client: Anthropic, kb: KnowledgeBase, question: str) -> str:
    hits = kb.search(question)
    context = "\n\n".join(f"[{src}]\n{text}" for src, text in hits) or "No relevant documents."
    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def main() -> None:
    client = Anthropic()
    kb = KnowledgeBase(KB_DIR)
    print(f"{BUSINESS_NAME} agent ready ({len(kb.items)} chunks). Type 'exit' to quit.\n")
    while True:
        q = input("Customer: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            print(f"\nAgent: {answer(client, kb, q)}\n")


if __name__ == "__main__":
    main()
