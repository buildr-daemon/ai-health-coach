"""
LangGraph-based Health Agent with RAG, memory extraction, and context management.
Handles long-term memory and medical protocol retrieval.
"""
from typing import List, TypedDict, Annotated, Optional, Any
import operator
import json
import logging
import os

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2")
LLM_TEMPERATURE_FACTUAL = 0.1  # For RAG/extraction - need accuracy
LLM_TEMPERATURE_CONVERSATIONAL = 0.4  # For responses - need warmth
MAX_CONTEXT_MESSAGES = 4  # Messages to keep in active context (last 4)
SUMMARIZE_EVERY_N_MESSAGES = 4  # Summarize every 4 new messages


# === State Definition ===

class AgentState(TypedDict):
    """
    Simplified state for the health agent graph.
    """
    # User identification
    user_id: int
    user_display_name: Optional[str]
    user_age: Optional[int]
    user_biological_sex: Optional[str]
    
    # Conversation messages (append-only through operator.add)
    messages: Annotated[List[dict], operator.add]
    
    # Long-term context
    user_memory_facts: List[dict]  # List of {fact_content, category}
    
    # RAG context
    retrieved_protocols: List[dict]  # List of {title, content, severity}
    
    # Rolling summary (replaces old messages beyond last 4)
    rolling_summary: Optional[str]
    
    # Output
    final_response: str
    extracted_facts: List[dict]  # New facts to save


# === Helper Functions ===

def extract_keywords_from_message(message_content: str) -> List[str]:
    """
    Extract potential medical keywords from a message for RAG matching.
    Uses simple pattern matching - could be enhanced with NER.
    """
    symptom_keywords = [
        "fever", "temperature", "hot", "cold", "chill",
        "headache", "head", "migraine", "pain",
        "stomach", "belly", "nausea", "vomit", "diarrhea", "constipation",
        "cough", "cold", "flu", "sneeze", "congestion", "throat",
        "back", "spine", "muscle", "ache",
        "allergy", "allergic", "hives", "itch", "rash", "skin",
        "sleep", "insomnia", "tired", "fatigue", "exhausted",
        "stress", "anxiety", "anxious", "worried", "panic", "nervous",
        "burn", "cut", "wound", "injury",
        "dizzy", "dizziness", "faint", "weak",
        "breathing", "breath", "chest",
        "refund", "policy", "support", "cancel"
    ]
    
    message_lower = message_content.lower()
    found_keywords = []
    
    for keyword in symptom_keywords:
        if keyword in message_lower:
            found_keywords.append(keyword)
    
    return found_keywords


def match_protocols_by_keywords(
    keywords: List[str], 
    protocols: List[Any],
    max_results: int = 3
) -> List[dict]:
    """
    Match protocols based on keyword overlap.
    Returns protocols sorted by relevance score.
    """
    scored_protocols = []
    
    for protocol in protocols:
        if not protocol.is_active:
            continue
            
        protocol_keywords = (protocol.keywords or "").lower().split(",")
        protocol_keywords = [k.strip() for k in protocol_keywords]
        
        # Calculate overlap score
        matches = set(keywords) & set(protocol_keywords)
        if matches:
            score = len(matches) / len(keywords) if keywords else 0
            scored_protocols.append({
                "protocol_id": protocol.id,
                "title": protocol.title,
                "content": protocol.content.strip(),
                "severity_level": protocol.severity_level,
                "requires_professional_followup": protocol.requires_professional_followup,
                "relevance_score": score,
                "matched_keywords": list(matches)
            })
    
    scored_protocols.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored_protocols[:max_results]


# === Database Interaction Functions ===

def retrieve_protocols_from_db(
    db_session: Session, 
    keywords: List[str],
    max_results: int = 3
) -> List[dict]:
    """
    Retrieve relevant medical protocols from database based on keywords.
    """
    from db.models import MedicalProtocol
    
    if not keywords:
        return []
    
    protocols = db_session.query(MedicalProtocol).filter(
        MedicalProtocol.is_active == True
    ).all()
    
    return match_protocols_by_keywords(keywords, protocols, max_results)


def retrieve_user_memories_from_db(
    db_session: Session, 
    user_id: int,
    limit: int = 10
) -> List[dict]:
    """
    Retrieve user's long-term memory facts from database.
    Returns most recent active facts.
    """
    from db.models import UserMemoryFact
    
    memories = db_session.query(UserMemoryFact).filter(
        UserMemoryFact.user_id == user_id,
        UserMemoryFact.is_active == True
    ).order_by(
        UserMemoryFact.created_at.desc()
    ).limit(limit).all()
    
    return [
        {
            "fact_id": m.id,
            "fact_content": m.fact_content,
            "category": m.category.value if m.category else None
        }
        for m in memories
    ]


def get_rolling_summary_from_db(
    db_session: Session, 
    user_id: int
) -> Optional[str]:
    """
    Get the current rolling conversation summary.
    """
    from db.models import ConversationSummary
    
    summary = db_session.query(ConversationSummary).filter(
        ConversationSummary.user_id == user_id
    ).order_by(
        ConversationSummary.created_at.desc()
    ).first()
    
    return summary.summary_content if summary else None


def save_extracted_facts_to_db(
    db_session: Session,
    user_id: int,
    facts: List[dict],
    source_message_id: Optional[int] = None
) -> int:
    """
    Save extracted facts to user's long-term memory.
    Returns count of facts saved.
    """
    from db.models import UserMemoryFact, MemoryCategory
    
    saved_count = 0
    for fact in facts:
        category_str = fact.get("category", "").upper()
        try:
            category = MemoryCategory[category_str] if category_str else MemoryCategory.SYMPTOM
        except KeyError:
            category = MemoryCategory.SYMPTOM
        
        memory = UserMemoryFact(
            user_id=user_id,
            fact_content=fact["fact_text"],
            category=category,
            source_message_id=source_message_id,
            extraction_confidence=fact.get("confidence", 0.8),
            is_active=True
        )
        db_session.add(memory)
        saved_count += 1
    
    if saved_count > 0:
        db_session.commit()
    
    return saved_count


def save_rolling_summary_to_db(
    db_session: Session,
    user_id: int,
    summary_content: str,
    message_count: int
):
    """Save/update the rolling conversation summary."""
    from db.models import ConversationSummary
    
    # Delete old summaries and create new one (rolling replacement)
    db_session.query(ConversationSummary).filter(
        ConversationSummary.user_id == user_id
    ).delete()
    
    summary = ConversationSummary(
        user_id=user_id,
        summary_content=summary_content,
        covers_messages_from_id=0,
        covers_messages_to_id=0,
        message_count_summarized=message_count
    )
    db_session.add(summary)
    db_session.commit()


def get_user_profile(db_session: Session, user_id: int) -> dict:
    """Get user profile information."""
    from db.models import User
    
    user = db_session.query(User).filter(User.id == user_id).first()
    if not user:
        return {}
    
    return {
        "display_name": user.display_name,
        "age": user.age_years,
        "biological_sex": user.biological_sex
    }


def count_unsummarized_messages(db_session: Session, user_id: int) -> int:
    """Count messages that haven't been summarized yet."""
    from db.models import Message
    
    return db_session.query(Message).filter(
        Message.user_id == user_id,
        Message.is_summarized == False
    ).count()


def mark_messages_as_summarized(db_session: Session, user_id: int, message_ids: List[int]):
    """Mark messages as summarized."""
    from db.models import Message
    
    db_session.query(Message).filter(
        Message.user_id == user_id,
        Message.id.in_(message_ids)
    ).update({Message.is_summarized: True}, synchronize_session=False)
    db_session.commit()


# === Graph Nodes ===

def node_load_user_profile(state: AgentState, db_session: Session) -> dict:
    """
    Load user profile information.
    """
    logger.info(f"[Profile] Loading profile for user {state['user_id']}")
    
    profile = get_user_profile(db_session, state["user_id"])
    
    return {
        "user_display_name": profile.get("display_name"),
        "user_age": profile.get("age"),
        "user_biological_sex": profile.get("biological_sex")
    }


def node_retrieve_context(state: AgentState, db_session: Session) -> dict:
    """
    RAG Node: Retrieves relevant medical protocols and user memories.
    Also loads the rolling summary.
    """
    logger.info(f"[RAG] Retrieving context for user {state['user_id']}")
    
    # Get the last user message for keyword extraction
    user_messages = [m for m in state.get("messages", []) if m.get("role") == "user"]
    last_message_content = user_messages[-1]["content"] if user_messages else ""
    
    # Extract keywords for protocol matching
    keywords = extract_keywords_from_message(last_message_content)
    logger.info(f"[RAG] Extracted keywords: {keywords}")
    
    # Retrieve matching protocols
    protocols = retrieve_protocols_from_db(db_session, keywords)
    logger.info(f"[RAG] Retrieved {len(protocols)} protocols")
    
    # Retrieve user's long-term memories
    memories = retrieve_user_memories_from_db(db_session, state["user_id"])
    logger.info(f"[RAG] Retrieved {len(memories)} user memories")
    
    # Get rolling summary
    summary = get_rolling_summary_from_db(db_session, state["user_id"])
    
    return {
        "retrieved_protocols": protocols,
        "user_memory_facts": memories,
        "rolling_summary": summary
    }


def node_summarize_if_needed(state: AgentState, db_session: Session) -> dict:
    """
    Rolling Summary Handler: Summarize when we have 4+ unsummarized messages.
    Creates/updates a rolling summary that incorporates old content.
    """
    from db.models import Message
    
    # Get unsummarized messages count
    unsummarized_count = count_unsummarized_messages(db_session, state["user_id"])
    
    if unsummarized_count < SUMMARIZE_EVERY_N_MESSAGES:
        logger.info(f"[Summarize] Skipping - only {unsummarized_count} unsummarized messages")
        return {}
    
    logger.info(f"[Summarize] Triggering summarization, {unsummarized_count} unsummarized messages")
    
    # Get messages to summarize (all except last 4)
    all_messages = db_session.query(Message).filter(
        Message.user_id == state["user_id"],
        Message.is_summarized == False
    ).order_by(Message.id.asc()).all()
    
    # Keep last 4, summarize the rest
    if len(all_messages) <= MAX_CONTEXT_MESSAGES:
        return {}
    
    messages_to_summarize = all_messages[:-MAX_CONTEXT_MESSAGES]
    
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE_FACTUAL, base_url=OLLAMA_BASE_URL)
    
    existing_summary = state.get("rolling_summary", "")
    
    # Format messages for summarization
    messages_text = "\n".join([
        f"{m.role.value.upper()}: {m.content}" 
        for m in messages_to_summarize
    ])
    
    # Build prompt with proper handling of existing summary
    summary_section = ""
    if existing_summary:
        summary_section = f"Previous summary to incorporate:\n{existing_summary}\n\n"
    
    prompt = f"""Create a concise rolling summary of this health conversation.
Focus on: symptoms discussed, health concerns, lifestyle details, advice given, key decisions.
Keep it under 150 words. This will be used as context for future responses.

{summary_section}New messages to summarize:
{messages_text}

Rolling Summary:
"""
    
    try:
        response = llm.invoke(prompt)
        new_summary = response.content
        
        # Save rolling summary (replaces old one)
        save_rolling_summary_to_db(
            db_session,
            state["user_id"],
            new_summary,
            len(messages_to_summarize)
        )
        
        # Mark messages as summarized
        mark_messages_as_summarized(
            db_session, 
            state["user_id"], 
            [m.id for m in messages_to_summarize]
        )
        
        logger.info(f"[Summarize] Created rolling summary from {len(messages_to_summarize)} messages")
        return {"rolling_summary": new_summary}
        
    except Exception as e:
        logger.error(f"[Summarize] Error during summarization: {e}")
        return {}


def node_generate_response(state: AgentState) -> dict:
    """
    Main response generation node.
    Uses rolling summary + last 4 messages for context.
    """
    logger.info(f"[Generate] Generating response for user {state['user_id']}")
    
    llm = ChatOllama(
        model=LLM_MODEL_NAME, 
        temperature=LLM_TEMPERATURE_CONVERSATIONAL,
        base_url=OLLAMA_BASE_URL
    )
    
    # Build context sections
    user_context = _build_user_context(state)
    protocol_context = _build_protocol_context(state)
    memory_context = _build_memory_context(state)
    
    # Build system prompt
    system_prompt = f"""You are a friendly AI Health Coach named "Healthie". 
You chat like a real person on WhatsApp - warm, casual, helpful.

=== USER PROFILE ===
{user_context}

=== USER'S HEALTH HISTORY (from past conversations) ===
{memory_context}

=== RELEVANT MEDICAL GUIDELINES ===
{protocol_context}

=== CONVERSATION CONTEXT (Summary of earlier conversation) ===
{state.get('rolling_summary') or 'No prior conversation.'}

=== RESPONSE GUIDELINES ===
1. Be conversational - short paragraphs, natural language
2. Acknowledge their concern FIRST, then provide guidance
3. Use the medical protocols when relevant, but explain simply
4. If symptoms sound serious (severity 3+), recommend seeing a doctor
5. Never diagnose - you provide guidance, not medical diagnosis
6. Ask follow-up questions if symptoms are unclear
7. Remember and reference what they've told you before
8. End with a caring check-in or actionable next step
9. Use emojis sparingly and naturally (1-2 max per message)
10. Keep responses under 150 words for readability"""

    # Build messages for LLM (only last 4 messages)
    llm_messages = [SystemMessage(content=system_prompt)]
    
    # Add only the last 4 messages for context
    recent_messages = state.get("messages", [])[-MAX_CONTEXT_MESSAGES:]
    for msg in recent_messages:
        if msg["role"] == "user":
            llm_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            llm_messages.append(AIMessage(content=msg["content"]))
    
    try:
        response = llm.invoke(llm_messages)
        final_response = response.content
        logger.info(f"[Generate] Generated response: {final_response[:100]}...")
    except Exception as e:
        logger.error(f"[Generate] Error generating response: {e}")
        final_response = "I'm having a bit of trouble right now. Could you try sending that again? 🙏"
    
    return {"final_response": final_response}


def node_extract_facts(state: AgentState, db_session: Session) -> dict:
    """
    Extract health facts from the conversation for long-term memory.
    Only runs every 3 user messages to reduce API calls.
    """
    messages = state.get("messages", [])
    if len(messages) < 2:
        return {"extracted_facts": []}
    
    # Rate limiting: Only extract facts every 3 user messages
    user_messages = [m for m in messages if m.get("role") == "user"]
    if len(user_messages) % 3 != 0:
        logger.info(f"[Extract] Skipping fact extraction (rate limiting: {len(user_messages)} user messages)")
        return {"extracted_facts": []}
    
    # Get last few messages for extraction
    recent_messages = messages[-4:]
    
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE_FACTUAL, base_url=OLLAMA_BASE_URL)
    
    messages_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}" 
        for m in recent_messages
    ])
    
    prompt = f"""Extract any health-related facts about the user from this conversation.
Only extract clear, specific facts - not assumptions.

Categories:
- SYMPTOM: Current symptoms (e.g., "has headache for 2 days")
- LIFESTYLE: Habits (e.g., "exercises 3x per week", "works night shifts")
- MEDICAL_HISTORY: Past conditions (e.g., "has diabetes", "had surgery in 2020")
- MEDICATION: Current medications (e.g., "takes metformin daily")
- ALLERGY: Known allergies (e.g., "allergic to penicillin")
- PREFERENCE: Health preferences (e.g., "prefers natural remedies")
- DEMOGRAPHIC: Basic info (e.g., "is 35 years old", "is pregnant")

Conversation:
{messages_text}

Respond in JSON format:
{{"facts": [{{"fact_text": "...", "category": "SYMPTOM|LIFESTYLE|...", "confidence": 0.0-1.0}}]}}

If no clear facts, respond with {{"facts": []}}"""

    try:
        response = llm.invoke(prompt)
        
        # Parse JSON response
        response_text = response.content.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text)
        extracted_facts = result.get("facts", [])
        
        # Save facts to database
        if extracted_facts:
            saved = save_extracted_facts_to_db(
                db_session, 
                state["user_id"], 
                extracted_facts
            )
            logger.info(f"[Extract] Saved {saved} facts to memory")
        
        return {"extracted_facts": extracted_facts}
        
    except Exception as e:
        error_str = str(e)
        if "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
            logger.warning(f"[Extract] Quota limit reached, skipping fact extraction")
        else:
            logger.error(f"[Extract] Error extracting facts: {e}")
        return {"extracted_facts": []}


# === Context Building Helpers ===

def _build_user_context(state: AgentState) -> str:
    """Build user profile context string."""
    parts = []
    
    if state.get("user_display_name"):
        parts.append(f"Name: {state['user_display_name']}")
    else:
        parts.append("Name: Unknown")
    
    if state.get("user_age"):
        parts.append(f"Age: {state['user_age']} years")
    
    if state.get("user_biological_sex"):
        parts.append(f"Biological Sex: {state['user_biological_sex']}")
    
    return "\n".join(parts) if parts else "No profile information available."


def _build_protocol_context(state: AgentState) -> str:
    """Build medical protocols context string."""
    protocols = state.get("retrieved_protocols", [])
    
    if not protocols:
        return "No specific protocols matched. Use general health guidance."
    
    parts = []
    for p in protocols:
        severity_note = ""
        if p.get("severity_level", 1) >= 3:
            severity_note = " ⚠️ RECOMMEND DOCTOR CONSULTATION"
        if p.get("requires_professional_followup"):
            severity_note += " (Professional followup advised)"
            
        parts.append(f"""
[{p['title']}]{severity_note}
{p['content'][:500]}...
""")
    
    return "\n".join(parts)


def _build_memory_context(state: AgentState) -> str:
    """Build user memory facts context string."""
    memories = state.get("user_memory_facts", [])
    
    if not memories:
        return "No health history recorded yet."
    
    # Group by category
    by_category = {}
    for m in memories:
        cat = m.get("category", "other")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(m["fact_content"])
    
    parts = []
    for category, facts in by_category.items():
        parts.append(f"{category.title()}: {'; '.join(facts)}")
    
    return "\n".join(parts)


# === Main Agent Factory ===

def create_health_agent(db_session: Session):
    """
    Create a compiled health agent graph with database session injected.
    Simplified flow without onboarding (handled separately via form).
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes with db_session closure
    workflow.add_node(
        "load_profile",
        lambda state: node_load_user_profile(state, db_session)
    )
    workflow.add_node(
        "retrieve_context", 
        lambda state: node_retrieve_context(state, db_session)
    )
    workflow.add_node(
        "summarize_if_needed",
        lambda state: node_summarize_if_needed(state, db_session)
    )
    workflow.add_node(
        "generate_response",
        node_generate_response
    )
    workflow.add_node(
        "extract_facts",
        lambda state: node_extract_facts(state, db_session)
    )
    
    # Define flow (simplified - no onboarding)
    workflow.set_entry_point("load_profile")
    workflow.add_edge("load_profile", "retrieve_context")
    workflow.add_edge("retrieve_context", "summarize_if_needed")
    workflow.add_edge("summarize_if_needed", "generate_response")
    workflow.add_edge("generate_response", "extract_facts")
    workflow.add_edge("extract_facts", END)
    
    return workflow.compile()


def run_health_agent(
    db_session: Session,
    user_id: int,
    messages: List[dict]
) -> dict:
    """
    Main entry point: Run the health agent for a user message.
    
    Args:
        db_session: SQLAlchemy database session
        user_id: ID of the user
        messages: List of message dicts with 'role' and 'content'
    
    Returns:
        dict with 'final_response' and 'extracted_facts'
    """
    agent = create_health_agent(db_session)
    
    # Prepare initial state (simplified)
    initial_state = {
        "user_id": user_id,
        "user_display_name": None,
        "user_age": None,
        "user_biological_sex": None,
        "messages": messages,
        "user_memory_facts": [],
        "retrieved_protocols": [],
        "rolling_summary": None,
        "final_response": "",
        "extracted_facts": []
    }
    
    # Run the graph
    result = agent.invoke(initial_state)
    
    return {
        "final_response": result.get("final_response", "I'm not sure how to respond to that."),
        "extracted_facts": result.get("extracted_facts", [])
    }
