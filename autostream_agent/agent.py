"""
agent.py — AutoStream Conversational AI Agent (Complete Implementation)

Contains:
 - AgentState schema with CaptureStage sub-state machine
 - All LangGraph node functions
 - Prompt templates
 - mock_lead_capture tool
 - Graph assembly with SqliteSaver persistence
 - CLI conversation loop (run directly: python agent.py)

Architecture:
  START → intent_classifier → (GREETING) → greeting_handler → END
                            → (PRODUCT_INQUIRY) → rag_retriever → END
                            → (HIGH_INTENT_LEAD) → lead_capture_router
                                                 → ask_for_name → END
                                                 → ask_for_email → END
                                                 → ask_for_platform → END
                                                 → execute_lead_capture → END
"""

import os
import re
import json
import time
import sqlite3
import logging
from enum import Enum
from typing import Optional, List, Annotated
from typing_extensions import TypedDict
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
MODEL_NAME    = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
CHROMA_DIR    = os.path.join(BASE_DIR, "chroma_db")
CHECKPOINT_DB = os.path.join(BASE_DIR, "checkpoints.db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "autostream_kb"

# ── State Schema ──────────────────────────────────────────────────────────────

class CaptureStage(str, Enum):
    NOT_STARTED        = "NOT_STARTED"
    COLLECTING_NAME    = "COLLECTING_NAME"
    COLLECTING_EMAIL   = "COLLECTING_EMAIL"
    COLLECTING_PLATFORM = "COLLECTING_PLATFORM"
    COMPLETE           = "COMPLETE"


class LeadData(TypedDict, total=False):
    name:          Optional[str]
    email:         Optional[str]
    platform:      Optional[str]
    capture_stage: str


class AgentState(TypedDict, total=False):
    messages:          Annotated[list, add_messages]
    intent_history:    List[str]
    current_intent:    str
    intent_confidence: float
    retrieval_context: str
    lead_data:         LeadData
    turn_count:        int
    cta_shown:         bool
    lead_captured:     bool
    lead_log:          str          # stores mock_lead_capture console output


# ── Prompts ───────────────────────────────────────────────────────────────────

PERSONA = """You are AutoStream's friendly AI sales assistant.
AutoStream is an AI-powered video editing and streaming platform for content creators.

Personality rules:
- Helpful, warm, and enthusiastic about video creation
- Conversational but professional
- NEVER pushy — helpful first, sales second
- Concise: keep responses under 120 words unless asked for detail
- Use 1-2 emojis per response, sparingly
- NEVER invent prices, features, or policies not explicitly provided to you
- If information is not in your context, honestly say so"""


# Compact intent prompt (~100 tokens vs original ~600)
INTENT_CLASSIFIER_PROMPT = """Classify the message into ONE intent. Reply with ONLY valid JSON.

GREETING: hello/bye/disinterest/small talk
HIGH_INTENT_LEAD: wants to sign up/buy/start/create account  
PRODUCT_INQUIRY: asking about features/pricing/plans/policies (default)

Rules: confidence<0.6 → PRODUCT_INQUIRY

{"intent":"GREETING"|"PRODUCT_INQUIRY"|"HIGH_INTENT_LEAD","confidence":0.0-1.0,"reasoning":"one sentence"}"""

# Keywords for fast pre-classification (no LLM call needed)
_HIGH_INTENT_KWS = [
    "sign up", "sign me up", "i want to sign", "i want to buy", "i want to try",
    "i'd like to purchase", "get me started", "how do i get started", "let's do it",
    "i'm ready to buy", "create account", "create an account", "subscribe",
    "i want to create", "i want to start",
]
_GREETING_KWS = {
    "hi", "hello", "hey", "hola", "howdy", "yo", "sup",
    "bye", "goodbye", "cya", "later",
    "never mind", "nevermind", "not interested", "no thanks",
    "ok thanks", "that's fine", "forget it", "no need",
}


def _keyword_classify(msg: str) -> Optional[str]:
    """Fast keyword pre-filter — avoids LLM call for obvious cases."""
    m = msg.lower().strip()
    # High intent: check phrases
    for kw in _HIGH_INTENT_KWS:
        if kw in m:
            return "HIGH_INTENT_LEAD"
    # Greeting: check for very short messages or known greeting words
    words = m.split()
    if len(words) <= 4 and any(w in _GREETING_KWS for w in words):
        return "GREETING"
    if m in _GREETING_KWS:
        return "GREETING"
    return None  # fall through to LLM


def _invoke_with_retry(llm, messages, max_retries: int = 3, base_wait: float = 6.0):
    """Invoke LLM with exponential back-off on 429 rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit_exceeded" in err:
                if attempt < max_retries - 1:
                    wait = base_wait * (attempt + 1)  # 6s, 12s
                    logger.warning("Rate limited — waiting %.0fs (attempt %d/%d)", wait, attempt + 1, max_retries)
                    time.sleep(wait)
                    continue
            raise
    return llm.invoke(messages)  # final attempt


RAG_RESPONSE_PROMPT = """You are AutoStream's AI support assistant. Answer the user's question using ONLY the provided context below.

CONTEXT (retrieved from AutoStream knowledge base):
{context}

CRITICAL RULES — YOU MUST FOLLOW THESE:
1. Answer ONLY from the provided context above. Never use your training knowledge about pricing, features, or policies.
2. If the answer is not in the context, say: "I don't have that specific information right now. Reach out to support@autostream.ai for details."
3. Always cite which source your answer came from (e.g., "According to our Pro plan details...").
4. Keep response under 150 words.
5. Answer ONLY from the provided context — never invent information.

{cta_nudge}"""


GREETING_PROMPT = """You are AutoStream's friendly AI assistant.
The user is greeting you or making small talk. Welcome them warmly.

Your response MUST:
1. Welcome the user in AutoStream's brand voice (warm, enthusiastic about video creation)
2. Give a ONE sentence description of what AutoStream does
3. End with a soft, non-pushy CTA like "Want to know about our plans?" or "What can I help you with?"

Keep it under 60 words. Use 1 emoji."""


ASK_NAME_PROMPT = """You are AutoStream's friendly sales assistant.
A user wants to sign up for AutoStream. Ask for their first and last name.

Rules:
- Sound conversational and enthusiastic, not like a form
- Frame it as "getting them set up" or "personalizing their account"
- Keep it under 40 words
- Use 1 emoji
- Example: "Awesome, let's get you set up! 🎉 What's your name?"""


ASK_EMAIL_PROMPT = """You are AutoStream's friendly sales assistant.
You just got the user's name: {name}

Ask for their email address to set up their AutoStream account.
Rules:
- Use their name naturally in the response
- Frame it as "creating their account" not "collecting data"
- Keep it under 40 words
- Example: "Nice to meet you, {name}! 😊 What email should we use to set up your account?\""""


ASK_PLATFORM_PROMPT = """You are AutoStream's friendly sales assistant.
You have the user's name ({name}) and email ({email}).

Ask what platform they primarily create content on.
Rules:
- Sound like you're personalizing their experience
- List a few options: YouTube, Instagram, TikTok, Twitch, Facebook, LinkedIn, or other
- Keep it under 50 words
- Example: "Almost there! 🎬 What platform do you mainly stream or post on? YouTube, TikTok, Twitch, Instagram, or something else?\""""


LEAD_SUCCESS_PROMPT = """You are AutoStream's AI assistant. A new user just signed up!

User details:
- Name: {name}
- Email: {email}  
- Platform: {platform}

Write a warm, personalized confirmation message (under 80 words) that:
1. Congratulates them by name
2. Mentions their platform specifically
3. Tells them to check their email ({email}) for next steps
4. Expresses excitement about having them join AutoStream
Use 2 emojis. Make it feel genuine, not robotic."""


SUMMARIZE_PROMPT = """Summarize the following conversation history in 2-3 sentences. 
Capture the main topics discussed and any key information shared.
This summary will replace the early messages to keep the context window manageable.

Conversation:
{history}

Summary:"""


# ── Mock Tool ─────────────────────────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulate capturing a lead in the CRM.
    In production: POST to HubSpot / Salesforce.
    For demo: logs to console and returns success dict.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "status": "success",
        "timestamp": timestamp,
        "data": {"name": name, "email": email, "platform": platform},
    }
    # This console output is what the demo video must show
    print("\n" + "=" * 55)
    print("  🎯  LEAD CAPTURED — mock_lead_capture() FIRED")
    print("=" * 55)
    print(f"  Name:      {name}")
    print(f"  Email:     {email}")
    print(f"  Platform:  {platform}")
    print(f"  Timestamp: {timestamp}")
    print("=" * 55 + "\n")
    return result


# ── LLM & Embeddings ──────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
        temperature=temperature,
    )


_embeddings_instance = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance


def get_retriever(k: int = 3):
    """MMR retriever — balances relevance with diversity."""
    vs = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "lambda_mult": 0.7},
    )


# ── Helper: Email regex ───────────────────────────────────────────────────────

EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')

KNOWN_PLATFORMS = [
    "youtube", "instagram", "tiktok", "twitch", "facebook",
    "linkedin", "twitter", "x", "kick", "vimeo", "snapchat",
]


# ── Pydantic model for bulk extraction ───────────────────────────────────────

class LeadExtraction(BaseModel):
    name:     str = Field(default="", description="Full name of the user, empty if not present")
    email:    str = Field(default="", description="Email address, empty if not present")
    platform: str = Field(default="", description="Streaming platform name, empty if not present")


def _bulk_extract(msg: str, lead_data: dict) -> dict:
    """Try to extract name/email/platform all at once from a single message."""
    if not msg:
        return lead_data
    try:
        llm_struct = get_llm(0).with_structured_output(LeadExtraction)
        res = llm_struct.invoke([
            SystemMessage(content="Extract the user's name, email address, and streaming platform from the message. Return empty strings for any field not mentioned."),
            HumanMessage(content=msg),
        ])
        new_data = dict(lead_data)
        if res.name and not new_data.get("name"):
            new_data["name"] = res.name.strip().title()
        if res.email:
            m = EMAIL_RE.search(res.email)
            if m and not new_data.get("email"):
                new_data["email"] = m.group(0).lower()
        if res.platform and not new_data.get("platform"):
            new_data["platform"] = res.platform.strip().title()
        return new_data
    except Exception as e:
        logger.warning("Bulk extraction failed: %s", e)
        return lead_data


def _inline_rag(question: str) -> str:
    """Lightweight inline RAG for mid-flow product questions."""
    try:
        llm = get_llm(0.3)
        rewrite_resp = llm.invoke([
            SystemMessage(content="Rewrite this question as a clean retrieval query in 5-8 words:"),
            HumanMessage(content=question),
        ])
        query = rewrite_resp.content.strip()
        docs = get_retriever(k=3).invoke(query)
        if not docs:
            return "I don't have that info right now. I'll follow up after we get you set up!"
        context = "\n\n".join([f"[{d.metadata.get('source','?')}] {d.page_content}" for d in docs])
        resp = llm.invoke([
            SystemMessage(content=RAG_RESPONSE_PROMPT.format(context=context, cta_nudge="")),
            HumanMessage(content=question),
        ])
        return resp.content
    except Exception as e:
        logger.warning("Inline RAG failed: %s", e)
        return "I'm having a moment — let me get back to making sure your account is set up!"


def _advance_stage(lead_data: dict) -> str:
    """Determine correct capture stage based on which fields are filled."""
    name     = lead_data.get("name")
    email    = lead_data.get("email")
    platform = lead_data.get("platform")
    if name and email and platform:
        return CaptureStage.COMPLETE.value
    if name and email:
        return CaptureStage.COLLECTING_PLATFORM.value
    if name:
        return CaptureStage.COLLECTING_EMAIL.value
    return CaptureStage.COLLECTING_NAME.value


def _get_latest_human_msg(state: AgentState) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content if hasattr(msg, "content") else str(msg)
    return ""


def _conversation_summary(state: AgentState) -> str:
    """Generate summary of early messages (for turn > 5 context compression)."""
    messages = state.get("messages", [])
    if len(messages) < 6:
        return ""
    early = messages[:-4]
    history = "\n".join([
        f"{'User' if getattr(m,'type','')=='human' else 'Agent'}: {getattr(m,'content','')}"
        for m in early
    ])
    try:
        resp = get_llm(0).invoke([
            SystemMessage(content=SUMMARIZE_PROMPT.format(history=history))
        ])
        return resp.content.strip()
    except Exception:
        return ""


# ── Node: Intent Classifier ───────────────────────────────────────────────────

def intent_classifier(state: AgentState) -> dict:
    latest = _get_latest_human_msg(state)
    intent_history = state.get("intent_history", [])
    turn_count = state.get("turn_count", 0) + 1
    confidence = 0.95

    # ── Fast path: keyword pre-filter (no LLM call) ───────────────────────────
    intent = _keyword_classify(latest)

    # ── Slow path: LLM classification (compact prompt, with retry) ────────────
    if intent is None:
        try:
            llm = get_llm(0)
            resp = _invoke_with_retry(llm, [
                SystemMessage(content=INTENT_CLASSIFIER_PROMPT),
                HumanMessage(content=latest),
            ])
            raw = resp.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            intent = data.get("intent", "PRODUCT_INQUIRY")
            confidence = float(data.get("confidence", 0.5))
            if confidence < 0.6:
                intent = "PRODUCT_INQUIRY"
        except Exception as e:
            logger.warning("Intent classification failed: %s", e)
            intent = "PRODUCT_INQUIRY"
            confidence = 0.5

    new_history = (intent_history + [intent])[-6:]

    updates = {
        "current_intent": intent,
        "intent_confidence": confidence,
        "intent_history": new_history,
        "turn_count": turn_count,
    }
    return updates


def route_by_intent(state: AgentState) -> str:
    """Conditional edge: route from intent_classifier to the right handler."""
    intent       = state.get("current_intent", "PRODUCT_INQUIRY")
    lead_data    = state.get("lead_data", {})
    capture_stage = lead_data.get("capture_stage", CaptureStage.NOT_STARTED.value)

    # If mid-flow lead capture, stay in flow unless user explicitly cancels
    if capture_stage not in (CaptureStage.NOT_STARTED.value, CaptureStage.COMPLETE.value):
        latest = _get_latest_human_msg(state).lower()
        exit_kws = ["stop", "cancel", "never mind", "nevermind", "not interested", "forget it", "no thanks"]
        if any(kw in latest for kw in exit_kws):
            logger.info("Graceful exit from lead capture → greeting_handler")
            return "greeting_handler"
        return "lead_capture_router"

    if intent == "GREETING":
        return "greeting_handler"
    if intent == "HIGH_INTENT_LEAD":
        return "lead_capture_router"
    return "rag_retriever"


# ── Node: Greeting Handler ────────────────────────────────────────────────────

def greeting_handler(state: AgentState) -> dict:
    lead_data     = state.get("lead_data", {})
    capture_stage = lead_data.get("capture_stage", CaptureStage.NOT_STARTED.value)

    try:
        llm = get_llm(0.7)
        resp = llm.invoke([
            SystemMessage(content=GREETING_PROMPT),
            HumanMessage(content=_get_latest_human_msg(state) or "Hello"),
        ])
        response = resp.content
    except Exception:
        response = "Hey there! 👋 Welcome to AutoStream — your AI-powered video creation platform. Want to know about our plans or features?"

    updates = {"messages": [AIMessage(content=response)]}

    # Graceful exit: if user greeted during lead capture, reset stage
    if capture_stage not in (CaptureStage.NOT_STARTED.value, CaptureStage.COMPLETE.value):
        updates["lead_data"] = {**lead_data, "capture_stage": CaptureStage.NOT_STARTED.value}

    return updates


# ── Node: RAG Retriever ───────────────────────────────────────────────────────

def rag_retriever(state: AgentState) -> dict:
    question      = _get_latest_human_msg(state)
    intent_history = state.get("intent_history", [])
    cta_shown     = state.get("cta_shown", False)

    # Step 1: Query rewriting — only for longer/complex questions to save tokens
    search_query = question
    if len(question.split()) > 8:
        try:
            llm = get_llm(0)
            rewrite_resp = _invoke_with_retry(llm, [
                SystemMessage(content="Rewrite this question as a 5-8 word retrieval query. Return ONLY the query."),
                HumanMessage(content=question),
            ])
            search_query = rewrite_resp.content.strip()
        except Exception:
            search_query = question

    # Step 2: MMR retrieval — k=3, diverse results
    try:
        retriever = get_retriever(k=3)
        docs = retriever.invoke(search_query)
        if docs:
            context_parts = [
                f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}"
                for d in docs
            ]
            context = "\n\n".join(context_parts)
            sources = list({d.metadata.get("source", "unknown") for d in docs})
        else:
            context = ""
            sources = []
    except Exception as e:
        logger.warning("Retrieval failed: %s", e)
        context = ""
        sources = []

    # Step 3: Soft CTA when 3+ consecutive PRODUCT_INQUIRY turns
    cta_nudge = ""
    consecutive_pq = sum(1 for i in intent_history[-3:] if i == "PRODUCT_INQUIRY")
    if consecutive_pq >= 3 and not cta_shown:
        cta_nudge = "\nEnd your response with: 'Ready to get started? Just say the word! 🚀'"

    # Step 4: Grounded response generation
    if not context:
        response = "I don't have specific information about that right now. Reach out to support@autostream.ai for details — our team is happy to help! 😊"
    else:
        try:
            llm = get_llm(0.3)
            resp = _invoke_with_retry(llm, [
                SystemMessage(content=RAG_RESPONSE_PROMPT.format(context=context, cta_nudge=cta_nudge)),
                HumanMessage(content=question),
            ])
            response = resp.content
            # Append source citation
            if sources:
                src_str = ", ".join(s.replace("_", " ").title() for s in sources)
                response += f"\n\n*Source: {src_str}*"
        except Exception:
            response = "I'm having a moment — could you try asking again? You can also reach support@autostream.ai 🙏"

    updates: dict = {
        "messages": [AIMessage(content=response)],
        "retrieval_context": context,
    }
    if cta_nudge:
        updates["cta_shown"] = True
    return updates


# ── Node: Lead Capture Router ─────────────────────────────────────────────────

def lead_capture_router(state: AgentState) -> dict:
    """
    Pure routing node — no response generated.
    Enforces explicit qualification flow order:
    Name -> Email -> Platform -> Execute lead capture.
    """
    lead_data     = state.get("lead_data") or {}
    lead_captured = state.get("lead_captured", False)

    if lead_captured:
        return {
            "messages": [AIMessage(content="You're already all set! 🎉 Is there anything else I can help you with?")]
        }

    # Initialise stage if fresh entry
    current_stage = lead_data.get("capture_stage", CaptureStage.NOT_STARTED.value)
    if current_stage == CaptureStage.NOT_STARTED.value:
        return {
            "lead_data": {**lead_data, "capture_stage": CaptureStage.COLLECTING_NAME.value}
        }
    return {}


def route_lead_capture(state: AgentState) -> str:
    """Conditional edge after lead_capture_router."""
    lead_data     = state.get("lead_data", {})
    lead_captured = state.get("lead_captured", False)
    stage         = lead_data.get("capture_stage", CaptureStage.NOT_STARTED.value)

    if lead_captured:
        return "greeting_handler"

    routes = {
        CaptureStage.NOT_STARTED.value:         "ask_for_name",
        CaptureStage.COLLECTING_NAME.value:     "ask_for_name",
        CaptureStage.COLLECTING_EMAIL.value:    "ask_for_email",
        CaptureStage.COLLECTING_PLATFORM.value: "ask_for_platform",
        CaptureStage.COMPLETE.value:            "execute_lead_capture",
    }
    return routes.get(stage, "ask_for_name")


# ── Node: Ask for Name ────────────────────────────────────────────────────────

def ask_for_name(state: AgentState) -> dict:
    lead_data      = state.get("lead_data", {})
    current_stage  = lead_data.get("capture_stage", "")
    latest         = _get_latest_human_msg(state)
    current_intent = state.get("current_intent", "")

    # Check if likely answering the name prompt
    words = latest.strip().split() if latest else []
    looks_like_name = 1 <= len(words) <= 5 and "@" not in latest and "?" not in latest

    # Inline RAG if mid-flow product question
    rag_prefix = ""
    if current_intent == "PRODUCT_INQUIRY" and latest and not looks_like_name:
        rag_prefix = _inline_rag(latest) + "\n\nNow, back to getting you set up — "

    # Extract name from response if we're collecting it
    if current_stage == CaptureStage.COLLECTING_NAME.value and latest and (current_intent != "PRODUCT_INQUIRY" or looks_like_name):
        # Heuristic: short message without @ or ? is likely a name
        if looks_like_name:
            name = latest.strip().title()
            return {
                "lead_data": {**lead_data, "name": name, "capture_stage": CaptureStage.COLLECTING_EMAIL.value},
                "messages": [AIMessage(content=_ask_email(name))],
            }

    # Generate the "what's your name?" question
    try:
        resp = get_llm(0.7).invoke([
            SystemMessage(content=ASK_NAME_PROMPT),
            HumanMessage(content="I want to sign up"),
        ])
        question = resp.content
    except Exception:
        question = "Awesome, let's get you set up! 🎉 What's your name?"

    return {
        "lead_data": {**lead_data, "capture_stage": CaptureStage.COLLECTING_NAME.value},
        "messages": [AIMessage(content=rag_prefix + question if rag_prefix else question)],
    }


def _ask_email(name: str) -> str:
    try:
        resp = get_llm(0.7).invoke([
            SystemMessage(content=ASK_EMAIL_PROMPT.format(name=name)),
            HumanMessage(content=f"My name is {name}"),
        ])
        return resp.content
    except Exception:
        return f"Nice to meet you, {name}! 😊 What email should we use to set up your AutoStream account?"


def _ask_platform(name: str, email: str) -> str:
    try:
        resp = get_llm(0.7).invoke([
            SystemMessage(content=ASK_PLATFORM_PROMPT.format(name=name, email=email)),
            HumanMessage(content=f"My email is {email}"),
        ])
        return resp.content
    except Exception:
        return f"Almost there! 🎬 What platform do you mainly create on? YouTube, TikTok, Twitch, Instagram, or something else?"


# ── Node: Ask for Email ───────────────────────────────────────────────────────

def ask_for_email(state: AgentState) -> dict:
    lead_data      = state.get("lead_data", {})
    name           = lead_data.get("name", "there")
    current_stage  = lead_data.get("capture_stage", "")
    latest         = _get_latest_human_msg(state)
    current_intent = state.get("current_intent", "")

    match = EMAIL_RE.search(latest) if latest else None
    looks_like_email = bool(match)

    # Inline RAG if mid-flow question
    rag_prefix = ""
    if current_intent == "PRODUCT_INQUIRY" and latest and not looks_like_email:
        rag_prefix = _inline_rag(latest) + "\n\nNow, back to getting you set up — "

    # Validate and extract email
    if current_stage == CaptureStage.COLLECTING_EMAIL.value and latest and (current_intent != "PRODUCT_INQUIRY" or looks_like_email):
        if looks_like_email and match:
            email = match.group(0).lower()
            return {
                "lead_data": {**lead_data, "email": email, "capture_stage": CaptureStage.COLLECTING_PLATFORM.value},
                "messages": [AIMessage(content=_ask_platform(name, email))],
            }
        else:
            # Validation failed — ask again with helpful error
            return {
                "messages": [AIMessage(
                    content=f"Hmm, that doesn't look like a valid email. Could you try again? "
                            f"It should look like: name@example.com 📧"
                )]
            }

    # Generate email ask
    question = _ask_email(name)
    return {
        "lead_data": {**lead_data, "capture_stage": CaptureStage.COLLECTING_EMAIL.value},
        "messages": [AIMessage(content=rag_prefix + question if rag_prefix else question)],
    }


# ── Node: Ask for Platform ────────────────────────────────────────────────────

def ask_for_platform(state: AgentState) -> dict:
    lead_data      = state.get("lead_data", {})
    name           = lead_data.get("name", "there")
    email          = lead_data.get("email", "")
    current_stage  = lead_data.get("capture_stage", "")
    latest         = _get_latest_human_msg(state)
    current_intent = state.get("current_intent", "")

    msg_lower = latest.lower() if latest else ""
    detected = next((p.title() for p in KNOWN_PLATFORMS if p in msg_lower), None)
    if detected is None and latest:
        words = latest.strip().split()
        if 1 <= len(words) <= 4:
            detected = latest.strip().title()
    looks_like_platform = bool(detected)

    # Inline RAG if mid-flow question
    rag_prefix = ""
    if current_intent == "PRODUCT_INQUIRY" and latest and not looks_like_platform:
        rag_prefix = _inline_rag(latest) + "\n\nNow, back to getting you set up — "

    # Validate and extract platform
    if current_stage == CaptureStage.COLLECTING_PLATFORM.value and latest and (current_intent != "PRODUCT_INQUIRY" or looks_like_platform):
        if detected:
            return {
                "lead_data": {**lead_data, "platform": detected, "capture_stage": CaptureStage.COMPLETE.value},
            }
        else:
            return {
                "messages": [AIMessage(
                    content="I didn't catch that platform. Could you tell me — is it YouTube, TikTok, Instagram, Twitch, Facebook, LinkedIn, or another? 🎬"
                )]
            }

    question = _ask_platform(name, email)
    return {
        "lead_data": {**lead_data, "capture_stage": CaptureStage.COLLECTING_PLATFORM.value},
        "messages": [AIMessage(content=rag_prefix + question if rag_prefix else question)],
    }


# ── Node: Execute Lead Capture ────────────────────────────────────────────────

def execute_lead_capture(state: AgentState) -> dict:
    """
    Guarded node — fires mock_lead_capture only after ALL 4 guards pass:
    Guard 1: capture_stage == COMPLETE
    Guard 2: name is present and non-empty
    Guard 3: email is present and non-empty
    Guard 4: lead_captured is False
    """
    lead_data     = state.get("lead_data", {})
    lead_captured = state.get("lead_captured", False)

    # ─ Guard 1: Idempotency ───────────────────────────────────────────────────
    if lead_captured:
        return {
            "messages": [AIMessage(content="You're already registered! 🎉 Is there anything else I can help with?")]
        }

    # ─ Guard 2: Stage gate ────────────────────────────────────────────────────
    stage = lead_data.get("capture_stage", "")
    if stage != CaptureStage.COMPLETE.value:
        return {}  # re-route

    # ─ Guard 3: Field validation ──────────────────────────────────────────────
    name     = lead_data.get("name", "")
    email    = lead_data.get("email", "")
    platform = lead_data.get("platform", "")

    if not name or not re.search(r'[a-zA-Z]', name):
        return {
            "messages": [AIMessage(content="Let me get your name first — what should I call you?")],
            "lead_data": {**lead_data, "capture_stage": CaptureStage.COLLECTING_NAME.value},
        }
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        return {
            "messages": [AIMessage(content=f"Could you double-check your email, {name}? It needs to be a valid email address.")],
            "lead_data": {**lead_data, "capture_stage": CaptureStage.COLLECTING_EMAIL.value},
        }
    if not platform:
        return {
            "messages": [AIMessage(content=f"Almost there, {name}! Which platform do you main? YouTube, TikTok, Twitch...?")],
            "lead_data": {**lead_data, "capture_stage": CaptureStage.COLLECTING_PLATFORM.value},
        }

    # ─ Guard 4: All clear — FIRE ─────────────────────────────────────────────
    try:
        result = mock_lead_capture(name=name, email=email, platform=platform)
        log_str = (
            f"✅ mock_lead_capture() fired successfully\n"
            f"   Name: {name} | Email: {email} | Platform: {platform}\n"
            f"   Status: {result.get('status')} | Time: {result.get('timestamp')}"
        )
    except Exception as e:
        return {
            "messages": [AIMessage(content="Something went wrong on our end — please reach out to support@autostream.ai 🙏")]
        }

    # Personalized confirmation message
    try:
        resp = get_llm(0.7).invoke([
            SystemMessage(content=LEAD_SUCCESS_PROMPT.format(name=name, email=email, platform=platform)),
            HumanMessage(content="Generate the confirmation message now."),
        ])
        confirmation = resp.content
    except Exception:
        confirmation = (
            f"🎉 You're officially in, {name}! Your AutoStream account is being set up now. "
            f"Check {email} for your welcome email with everything you need to start "
            f"creating amazing content on {platform}. Welcome to the AutoStream family! 🚀"
        )

    return {
        "messages": [AIMessage(content=confirmation)],
        "lead_captured": True,
        "lead_log": log_str,
    }


# ── Graph Builder ─────────────────────────────────────────────────────────────

_graph_instance = None


def get_graph():
    """Return shared compiled graph (singleton)."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = _build_graph()
    return _graph_instance


def _build_graph():
    workflow = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────────
    workflow.add_node("intent_classifier",  intent_classifier)
    workflow.add_node("greeting_handler",   greeting_handler)
    workflow.add_node("rag_retriever",      rag_retriever)
    workflow.add_node("lead_capture_router", lead_capture_router)
    workflow.add_node("ask_for_name",       ask_for_name)
    workflow.add_node("ask_for_email",      ask_for_email)
    workflow.add_node("ask_for_platform",   ask_for_platform)
    workflow.add_node("execute_lead_capture", execute_lead_capture)

    # ── Entry point ────────────────────────────────────────────────────────
    workflow.set_entry_point("intent_classifier")

    # ── Intent router ──────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "greeting_handler":    "greeting_handler",
            "rag_retriever":       "rag_retriever",
            "lead_capture_router": "lead_capture_router",
        }
    )

    # ── Greeting & RAG → END ───────────────────────────────────────────────
    workflow.add_edge("greeting_handler", END)
    workflow.add_edge("rag_retriever",    END)

    # ── Lead capture router → sub-flow ────────────────────────────────────
    workflow.add_conditional_edges(
        "lead_capture_router",
        route_lead_capture,
        {
            "ask_for_name":         "ask_for_name",
            "ask_for_email":        "ask_for_email",
            "ask_for_platform":     "ask_for_platform",
            "execute_lead_capture": "execute_lead_capture",
            "greeting_handler":     "greeting_handler",
        }
    )

    # ── Sub-flow → END ─────────────────────────────────────────────────────
    workflow.add_edge("ask_for_name",           END)
    workflow.add_edge("ask_for_email",          END)
    workflow.add_edge("ask_for_platform",       END)
    workflow.add_edge("execute_lead_capture",   END)

    # ── SqliteSaver checkpointer — persists state across turns ─────────────
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = workflow.compile(checkpointer=checkpointer)
    return graph


def get_initial_state() -> dict:
    return {
        "messages":          [],
        "intent_history":    [],
        "current_intent":    "",
        "intent_confidence": 0.0,
        "retrieval_context": "",
        "lead_data": {
            "name":          None,
            "email":         None,
            "platform":      None,
            "capture_stage": CaptureStage.NOT_STARTED.value,
        },
        "turn_count":    0,
        "cta_shown":     False,
        "lead_captured": False,
        "lead_log":      "",
    }


# ── CLI Conversation Loop ─────────────────────────────────────────────────────

def main():
    import uuid
    print("\n" + "=" * 55)
    print("  AutoStream AI Agent — CLI Mode")
    print("  Type 'quit' to exit, 'reset' for new session")
    print("=" * 55 + "\n")

    graph = get_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Session ID: {thread_id[:8]}...\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if user_input.lower() == "reset":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print(f"\n🔄 New session: {thread_id[:8]}...\n")
            continue

        state = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)

        # Get last AI message
        ai_msgs = [m for m in state.get("messages", []) if hasattr(m, "type") and m.type == "ai"]
        response = ai_msgs[-1].content if ai_msgs else "..."

        print(f"\nAgent [{state.get('current_intent','?')}]: {response}\n")

        # Show lead data progress
        ld = state.get("lead_data", {})
        stage = ld.get("capture_stage", "")
        if stage not in ("NOT_STARTED", "", None):
            name  = ld.get("name",  "—")
            email = ld.get("email", "—")
            plat  = ld.get("platform", "—")
            print(f"  📋 Lead: stage={stage} | name={name} | email={email} | platform={plat}\n")


if __name__ == "__main__":
    main()
