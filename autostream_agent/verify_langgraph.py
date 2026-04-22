"""
verify_langgraph.py — Quick diagnostic script for AutoStream AI Agent

Run this to confirm your environment is correctly set up before using agent.py:
    python verify_langgraph.py

Checks:
  1. Package installation (langgraph, langchain, langchain-groq, chromadb)
  2. Minimal graph compiles and executes
  3. State persistence with same thread_id across two turns
  4. ChromaDB retrieval works
"""

import sys
import os
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
all_ok = True


def check(label: str, ok: bool, detail: str = ""):
    global all_ok
    status = PASS if ok else FAIL
    if not ok:
        all_ok = False
    print(f"  {status}  {label}" + (f"\n         → {detail}" if detail else ""))


print("\n" + "=" * 55)
print("  AutoStream AI Agent — Environment Verification")
print("=" * 55)

# ── Check 1: Package imports ────────────────────────────────────────────────────
print("\n📦 [1/4] Checking package installations...")

packages = [
    ("langgraph",           "LangGraph"),
    ("langchain",           "LangChain"),
    ("langchain_groq",      "langchain-groq"),
    ("langchain_huggingface","langchain-huggingface"),
    ("langchain_chroma",    "langchain-chroma"),
    ("chromadb",            "ChromaDB"),
    ("sentence_transformers","sentence-transformers"),
    ("dotenv",              "python-dotenv"),
    ("streamlit",           "Streamlit"),
]

for module, name in packages:
    try:
        __import__(module)
        check(name, True)
    except ImportError as e:
        check(name, False, str(e))

# ── Check 2: Minimal graph compiles ────────────────────────────────────────────
print("\n🔧 [2/4] Checking minimal LangGraph graph compilation...")
try:
    import sqlite3
    from typing import Annotated
    from typing_extensions import TypedDict
    from langchain_core.messages import HumanMessage, AIMessage
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.sqlite import SqliteSaver

    class TestState(TypedDict, total=False):
        messages: Annotated[list, add_messages]
        counter: int

    def dummy_node(state):
        return {"messages": [AIMessage(content="pong")], "counter": state.get("counter",0) + 1}

    wf = StateGraph(TestState)
    wf.add_node("dummy", dummy_node)
    wf.set_entry_point("dummy")
    wf.add_edge("dummy", END)

    db_path = os.path.join(BASE_DIR, "_verify_test.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cp = SqliteSaver(conn)
    g = wf.compile(checkpointer=cp)
    check("StateGraph compiles with SqliteSaver", True)

    # ── Check 3: State persistence across two turns ─────────────────────────────
    print("\n💾 [3/4] Checking state persistence (2-turn test)...")
    tid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    s1 = g.invoke({"messages": [HumanMessage(content="ping")], "counter": 0}, config)
    c1 = s1.get("counter", 0)
    check("Turn 1 executes and counter = 1", c1 == 1, f"counter={c1}")

    s2 = g.invoke({"messages": [HumanMessage(content="ping again")]}, config)
    c2 = s2.get("counter", 0)
    check("Turn 2 resumes state (counter = 2 via same thread_id)", c2 == 2, f"counter={c2}")

    msgs = [m for m in s2.get("messages", []) if hasattr(m,"type") and m.type=="ai"]
    check("Messages persist across turns", len(msgs) >= 1, f"ai_msgs={len(msgs)}")

    conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)

except Exception as e:
    check("LangGraph graph/persistence", False, str(e))

# ── Check 4: ChromaDB retrieval ─────────────────────────────────────────────────
print("\n🔍 [4/4] Checking ChromaDB retrieval...")
chroma_dir = os.path.join(BASE_DIR, "chroma_db")
if not os.path.exists(chroma_dir):
    check("chroma_db/ exists", False, "Run: python ingest.py first")
else:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)

        emb = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vs = Chroma(persist_directory=chroma_dir, embedding_function=emb, collection_name="autostream_kb")
        count = vs._collection.count()
        check(f"ChromaDB loaded ({count} chunks)", count > 0, f"count={count}")

        results = vs.similarity_search("Pro plan pricing 4K unlimited", k=2)
        sources = [r.metadata.get("source","?") for r in results]
        check("Retrieval returns relevant docs", len(results) > 0, f"sources={sources}")
    except Exception as e:
        check("ChromaDB retrieval", False, str(e))

# ── Summary ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if all_ok:
    print("  ✅  All checks passed — agent is ready!")
    print("\n  Start the UI:  streamlit run streamlit_app.py")
    print("  CLI mode:      python agent.py")
else:
    print("  ⚠️  Some checks failed — see details above")
    print("\n  Common fixes:")
    print("    pip install -r requirements.txt")
    print("    python ingest.py")
print("=" * 55 + "\n")
