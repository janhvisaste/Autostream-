"""
ingest.py — One-time knowledge base setup for AutoStream AI Agent.

Run this ONCE before starting the agent:
    python ingest.py

What this does:
1. Loads knowledge_base/knowledge_base.json
2. Converts JSON data into readable text chunks
3. Attaches metadata (topic, plan name, source) to each chunk
4. Embeds all chunks with HuggingFace all-MiniLM-L6-v2 (local, free)
5. Stores embedded chunks in ChromaDB at ./chroma_db
"""

import json
import os
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "knowledge_base", "knowledge_base.json")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "autostream_kb"

# ── Section metadata ───────────────────────────────────────────────────────────
SECTION_META = {
    "basic_plan":       {"topic": "pricing",    "plan": "basic", "doc_type": "plan_details"},
    "pro_plan":         {"topic": "pricing",    "plan": "pro",   "doc_type": "plan_details"},
    "refund_policy":    {"topic": "policy",     "plan": "all",   "doc_type": "policy"},
    "support_policy":   {"topic": "support",    "plan": "all",   "doc_type": "policy"},
    "plan_comparison":  {"topic": "comparison", "plan": "all",   "doc_type": "comparison"},
    "company_overview": {"topic": "general",    "plan": "all",   "doc_type": "overview"},
}


def json_section_to_text(key: str, data: dict) -> str:
    """Convert a JSON section into a rich readable string for embedding."""
    lines = []
    for field, value in data.items():
        if field == "features" and isinstance(value, list):
            lines.append(f"Features: {', '.join(value)}")
        elif isinstance(value, bool):
            lines.append(f"{field.replace('_', ' ').title()}: {'Yes' if value else 'No'}")
        else:
            lines.append(f"{field.replace('_', ' ').title()}: {value}")
    return "\n".join(lines)


def load_and_chunk_documents() -> list[Document]:
    """Load knowledge_base.json and split into chunks with metadata."""
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for section_key, section_data in kb.items():
        meta = SECTION_META.get(section_key, {
            "topic": "general", "plan": "all", "doc_type": "unknown"
        })
        text = json_section_to_text(section_key, section_data) if isinstance(section_data, dict) else str(section_data)
        raw_chunks = splitter.split_text(text)

        for i, chunk in enumerate(raw_chunks):
            all_chunks.append(Document(
                page_content=chunk,
                metadata={
                    "source": section_key,
                    "topic": meta["topic"],
                    "plan": meta["plan"],
                    "doc_type": meta["doc_type"],
                    "chunk_index": i,
                }
            ))
        print(f"  ✓ {section_key}: {len(raw_chunks)} chunks")

    return all_chunks


def main():
    print("=" * 55)
    print("  AutoStream AI Agent — Knowledge Base Ingestion")
    print("=" * 55)

    # Clear any stale ChromaDB (different embedding model = garbage results)
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("  🗑  Cleared existing chroma_db\n")

    print("📚 Loading knowledge_base.json and chunking...")
    chunks = load_and_chunk_documents()
    print(f"\n  → {len(chunks)} total chunks from {len(SECTION_META)} sections")

    print(f"\n🔢 Embedding with {EMBEDDING_MODEL}  (runs locally, ~30 seconds)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"\n💾 Writing to ChromaDB at {CHROMA_DIR}...")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    # Verification queries
    print("\n🔍 Running retrieval verification...")
    verifications = [
        ("Pro plan price unlimited 4K",     ["pro_plan", "plan_comparison"]),
        ("refund policy 7 days",             ["refund_policy"]),
        ("24/7 support priority",            ["support_policy", "pro_plan"]),
        ("720p video resolution Basic",      ["basic_plan", "plan_comparison"]),
        ("AI automatic captions",            ["pro_plan", "plan_comparison"]),
    ]

    all_ok = True
    for query, expected in verifications:
        results = store.similarity_search(query, k=3)
        sources = [r.metadata.get("source", "") for r in results]
        hit = any(s in sources for s in expected)
        status = "✅" if hit else "❌"
        if not hit:
            all_ok = False
        print(f"  {status} '{query}' → {sources}")

    print("\n" + "=" * 55)
    if all_ok:
        print("  ✅ Ingestion COMPLETE — all retrieval checks passed")
        print("\n  Next steps:")
        print("    streamlit run streamlit_app.py")
        print("    — or —")
        print("    python agent.py  (CLI mode)")
    else:
        print("  ⚠️  Ingestion done but some retrieval checks failed")
        print("     Check knowledge_base.json content and retry")
    print("=" * 55)


if __name__ == "__main__":
    main()
