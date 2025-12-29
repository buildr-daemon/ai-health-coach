"""
LangGraph-based Health Agent with RAG, memory extraction, and context management.
Handles onboarding, long-term memory, and medical protocol retrieval.
"""
from typing import List, TypedDict, Annotated, Optional, Any
import operator
import re
import json
import logging
import os

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2")  # Local Ollama model
LLM_TEMPERATURE_FACTUAL = 0.1  # For RAG/extraction - need accuracy
LLM_TEMPERATURE_CONVERSATIONAL = 0.4  # For responses - need warmth
MAX_CONTEXT_MESSAGES = 4  # Messages to keep in active context
SUMMARIZE_THRESHOLD = 5  # When to trigger summarization


# === State Definition ===

class AgentState(TypedDict):
    """
    Complete state for the health agent graph.
    Tracks all context needed for response generation.
    """
    # User identification
    user_id: int
    user_display_name: Optional[str]
    
    # Conversation messages (append-only through operator.add)
    messages: Annotated[List[dict], operator.add]
    
    # Long-term context
    user_health_profile_summary: Optional[str]
    user_memory_facts: List[dict]  # List of {fact_content, category}
    
    # RAG context
    retrieved_protocols: List[dict]  # List of {title, content, severity}
    
    # Conversation management
    conversation_summary: Optional[str]  # Summary of old messages
    
    # Onboarding state
    is_onboarding_conversation: bool
    onboarding_questions_asked: int
    
    # Output
    final_response: str
    extracted_facts: List[dict]  # New facts to save


# === Helper Functions ===

def estimate_token_count(text: str) -> int:
    """Rough token estimation (4 chars per token average)."""
    return len(text) // 4


def extract_keywords_from_message(message_content: str) -> List[str]:
    """
    Extract potential medical keywords from a message for RAG matching.
    Uses simple pattern matching - could be enhanced with NER.
    """
    # Common symptom keywords to look for
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
    
    # Sort by relevance score descending
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
    
    # Get all active protocols
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


def get_conversation_summary_from_db(
    db_session: Session, 
    user_id: int
) -> Optional[str]:
    """
    Get the most recent conversation summary for context overflow handling.
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


def save_conversation_summary_to_db(
    db_session: Session,
    user_id: int,
    summary_content: str,
    from_message_id: int,
    to_message_id: int,
    message_count: int
):
    """Save a conversation summary for context overflow handling."""
    from db.models import ConversationSummary
    
    summary = ConversationSummary(
        user_id=user_id,
        summary_content=summary_content,
        covers_messages_from_id=from_message_id,
        covers_messages_to_id=to_message_id,
        message_count_summarized=message_count
    )
    db_session.add(summary)
    db_session.commit()


# === Graph Nodes ===

def node_retrieve_context(state: AgentState, db_session: Session) -> dict:
    """
    RAG Node: Retrieves relevant medical protocols and user memories.
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
    
    # Get existing conversation summary (for overflow handling)
    summary = get_conversation_summary_from_db(db_session, state["user_id"])
    
    return {
        "retrieved_protocols": protocols,
        "user_memory_facts": memories,
        "conversation_summary": summary
    }


def node_check_onboarding(state: AgentState, db_session: Session) -> dict:
    """
    Check if this is an onboarding conversation and set appropriate flags.
    """
    from db.models import User, OnboardingStatus
    
    user = db_session.query(User).filter(User.id == state["user_id"]).first()
    
    if not user:
        return {"is_onboarding_conversation": False}
    
    is_onboarding = user.onboarding_status != OnboardingStatus.COMPLETED
    
    return {
        "is_onboarding_conversation": is_onboarding,
        "user_display_name": user.display_name,
        "user_health_profile_summary": user.health_profile_summary
    }


def node_summarize_if_needed(state: AgentState, db_session: Session) -> dict:
    """
    Context Overflow Handler: Summarize old messages if context is too long.
    """
    messages = state.get("messages", [])
    
    if len(messages) < SUMMARIZE_THRESHOLD:
        return {}
    
    logger.info(f"[Summarize] Triggering summarization, {len(messages)} messages in context")
    
    # Take oldest messages to summarize
    messages_to_summarize = messages[:len(messages) - MAX_CONTEXT_MESSAGES]
    
    if not messages_to_summarize:
        return {}
    
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE_FACTUAL, base_url=OLLAMA_BASE_URL)
    
    existing_summary = state.get("conversation_summary", "")
    
    # Format messages for summarization
    messages_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}" 
        for m in messages_to_summarize
    ])
    
    prompt = f"""Summarize this health conversation concisely. 
Focus on: symptoms mentioned, health concerns, lifestyle details, any advice given.
Keep it under 200 words.

{'Previous summary: ' + existing_summary if existing_summary else ''}

New messages to incorporate:
{messages_text}

Summary:"""
    
    try:
        response = llm.invoke(prompt)
        new_summary = response.content
        
        # Save summary to database
        if messages_to_summarize:
            # This is simplified - in production, messages would have IDs
            save_conversation_summary_to_db(
                db_session,
                state["user_id"],
                new_summary,
                from_message_id=0,
                to_message_id=0,
                message_count=len(messages_to_summarize)
            )
        
        logger.info(f"[Summarize] Created summary of {len(messages_to_summarize)} messages")
        return {"conversation_summary": new_summary}
        
    except Exception as e:
        logger.error(f"[Summarize] Error during summarization: {e}")
        return {}


def node_generate_response(state: AgentState) -> dict:
    """
    Main response generation node.
    Uses all available context to generate a helpful, empathetic response.
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
    onboarding_instructions = _build_onboarding_instructions(state)
    
    # Build system prompt
    system_prompt = f"""You are a friendly AI Health Coach named "Healthie". 
You chat like a real person on WhatsApp - warm, casual, helpful.

{onboarding_instructions}

=== USER PROFILE ===
{user_context}

=== USER'S HEALTH HISTORY (from past conversations) ===
{memory_context}

=== RELEVANT MEDICAL GUIDELINES ===
{protocol_context}

=== CONVERSATION CONTEXT ===
{state.get('conversation_summary', 'No prior summary.')}

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

    # Build messages for LLM
    llm_messages = [SystemMessage(content=system_prompt)]
    
    # Add conversation messages (limited to recent)
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
    Runs in the background to not block response.
    """
    # Only extract if there's a meaningful conversation
    messages = state.get("messages", [])
    if len(messages) < 2:
        return {"extracted_facts": []}
    
    # Rate limiting: Only extract facts every 3 messages to reduce API calls
    user_messages = [m for m in messages if m.get("role") == "user"]
    if len(user_messages) % 3 != 0:
        logger.info(f"[Extract] Skipping fact extraction (rate limiting: {len(user_messages)} user messages)")
        return {"extracted_facts": []}
    
    # Get last few messages for extraction
    recent_messages = messages[-4:]  # Last 2 exchanges
    
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
        # Handle markdown code blocks
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
        # Check if it's a quota error
        if "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
            logger.warning(f"[Extract] Quota limit reached, skipping fact extraction")
        else:
            logger.error(f"[Extract] Error extracting facts: {e}")
        return {"extracted_facts": []}


def node_update_onboarding_status(state: AgentState, db_session: Session) -> dict:
    """
    Update user's onboarding status based on conversation progress.
    """
    if not state.get("is_onboarding_conversation"):
        return {}
    
    from db.models import User, OnboardingStatus
    
    # Check if we have enough information to complete onboarding
    user = db_session.query(User).filter(User.id == state["user_id"]).first()
    if not user:
        return {}
    
    # Simple heuristic: after 3 exchanges, mark onboarding as in progress
    # After user provides name, mark as complete
    messages = state.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == "user"]
    
    if len(user_messages) >= 1 and user.onboarding_status == OnboardingStatus.NOT_STARTED:
        user.onboarding_status = OnboardingStatus.IN_PROGRESS
        db_session.commit()
        logger.info(f"[Onboarding] User {user.id} status -> IN_PROGRESS")
    
    # Check if user mentioned their name
    if user.display_name is None:
        for msg in user_messages:
            # Simple name extraction - could be enhanced with NER
            content_lower = msg["content"].lower()
            if any(phrase in content_lower for phrase in ["my name is", "i'm ", "i am ", "call me"]):
                # Try to extract name - improved regex to avoid false positives
                name_match = re.search(
                    r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", 
                    msg["content"], 
                    re.IGNORECASE
                )
                if name_match:
                    extracted_name = name_match.group(1).strip().title()
                    # Validate the name (at least 2 chars, not common false positives)
                    invalid_names = ["having", "going", "feeling", "doing", "being", "coming"]
                    if len(extracted_name) >= 2 and extracted_name.lower() not in invalid_names:
                        user.display_name = extracted_name
                        user.onboarding_status = OnboardingStatus.COMPLETED
                        db_session.commit()
                        logger.info(f"[Onboarding] Extracted name: {user.display_name}, status -> COMPLETED")
                        break
    
    return {}


# === Context Building Helpers ===

def _build_user_context(state: AgentState) -> str:
    """Build user profile context string."""
    parts = []
    
    if state.get("user_display_name"):
        parts.append(f"Name: {state['user_display_name']}")
    else:
        parts.append("Name: Unknown (ask them!)")
    
    if state.get("user_health_profile_summary"):
        parts.append(f"Health Profile: {state['user_health_profile_summary']}")
    
    return "\n".join(parts) if parts else "New user - no profile yet."


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


def _build_onboarding_instructions(state: AgentState) -> str:
    """Build onboarding-specific instructions."""
    if not state.get("is_onboarding_conversation"):
        return ""
    
    return """
=== ONBOARDING MODE ===
This is a new user! Your goals for this conversation:
1. Welcome them warmly
2. Ask for their name naturally
3. Ask about any current health concerns
4. Let them know you're here to help with health questions

Be friendly and not overwhelming - just get to know them naturally through conversation.
Don't ask all questions at once - let it flow like a real chat.
"""


# === Main Agent Factory ===

def create_health_agent(db_session: Session):
    """
    Create a compiled health agent graph with database session injected.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes with db_session closure
    workflow.add_node(
        "retrieve_context", 
        lambda state: node_retrieve_context(state, db_session)
    )
    workflow.add_node(
        "check_onboarding",
        lambda state: node_check_onboarding(state, db_session)
    )
    workflow.add_node(
        "summarize_if_needed",
        lambda state: node_summarize_if_needed(state, db_session)
    )
    workflow.add_node(
        "generate_response",
        node_generate_response  # Doesn't need db
    )
    workflow.add_node(
        "extract_facts",
        lambda state: node_extract_facts(state, db_session)
    )
    workflow.add_node(
        "update_onboarding",
        lambda state: node_update_onboarding_status(state, db_session)
    )
    
    # Define flow
    workflow.set_entry_point("check_onboarding")
    workflow.add_edge("check_onboarding", "retrieve_context")
    workflow.add_edge("retrieve_context", "summarize_if_needed")
    workflow.add_edge("summarize_if_needed", "generate_response")
    workflow.add_edge("generate_response", "extract_facts")
    workflow.add_edge("extract_facts", "update_onboarding")
    workflow.add_edge("update_onboarding", END)
    
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
    
    # Prepare initial state
    initial_state = {
        "user_id": user_id,
        "user_display_name": None,
        "messages": messages,
        "user_health_profile_summary": None,
        "user_memory_facts": [],
        "retrieved_protocols": [],
        "conversation_summary": None,
        "is_onboarding_conversation": False,
        "onboarding_questions_asked": 0,
        "final_response": "",
        "extracted_facts": []
    }
    
    # Run the graph
    result = agent.invoke(initial_state)
    
    return {
        "final_response": result.get("final_response", "I'm not sure how to respond to that."),
        "extracted_facts": result.get("extracted_facts", [])
    }


# === Legacy compatibility (if old code references app_agent) ===
# Note: This is deprecated - use run_health_agent() instead
class LegacyAgentWrapper:
    """Wrapper for backwards compatibility with old code."""
    
    def invoke(self, inputs: dict) -> dict:
        from db.database import SessionLocal
        db = SessionLocal()
        try:
            messages = inputs.get("messages", [])
            # Convert from old format if needed
            formatted_messages = []
            for m in messages:
                if hasattr(m, 'role') and hasattr(m, 'content'):
                    formatted_messages.append({"role": m.role, "content": m.content})
                elif isinstance(m, dict):
                    formatted_messages.append(m)
            
            result = run_health_agent(
                db,
                inputs.get("user_id", 1),
                formatted_messages
            )
            return result
        finally:
            db.close()

app_agent = LegacyAgentWrapper()
