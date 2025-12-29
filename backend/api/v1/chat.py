"""
Chat API endpoints for the Health Agent.
Handles initialization, messaging, history pagination, and typing indicators.
"""
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
    BackgroundTasks
)
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import Optional
import logging

from db.models import (
    User,
    Message,
    UserMemoryFact,
    MedicalProtocol,
    TypingIndicator,
    OnboardingStatus,
    MessageRole,
    ConversationSummary
)
from lib.schema import (
    ChatInitializationRequest,
    ChatInitializationResponse,
    ChatMessageResponse,
    SendMessageRequest,
    SendMessageResponse,
    ChatHistoryPaginationRequest,
    ChatHistoryPaginationResponse,
    TypingIndicatorRequest,
    TypingIndicatorResponse,
    ErrorResponse,
    OnboardingStatus as OnboardingStatusEnum,
    MessageRole as MessageRoleEnum,
    TypingStatus
)
from db.database import get_db
from lib.agent import run_health_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

# === Constants ===
INITIAL_HISTORY_LIMIT = 20  # Messages to load on init
MAX_MESSAGE_LENGTH = 4000  # Character limit for messages
TYPING_INDICATOR_TTL_SECONDS = 5  # How long typing indicator is valid


# === Error Helpers ===

def raise_user_not_found(user_id: int):
    """Raise standardized user not found error."""
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "error_code": "USER_NOT_FOUND",
            "error_message": f"User with ID {user_id} does not exist",
            "details": {"user_id": user_id}
        }
    )


def raise_validation_error(field: str, message: str):
    """Raise standardized validation error."""
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "error_code": "VALIDATION_ERROR",
            "error_message": message,
            "details": {"field": field}
        }
    )


# === Helper Functions ===

def message_to_response(message: Message) -> ChatMessageResponse:
    """Convert database Message to API response format."""
    return ChatMessageResponse(
        message_id=message.id,
        role=MessageRoleEnum(message.role.value),
        content=message.content,
        created_at=message.created_at
    )


def get_or_create_user(
    db: Session, 
    device_identifier: Optional[str] = None
) -> tuple[User, bool]:
    """
    Get existing user by device ID or create new user.
    Returns (user, is_new_user).
    """
    is_new_user = False
    
    if device_identifier:
        # Try to find existing user by device
        existing_user = db.query(User).filter(
            User.device_identifier == device_identifier
        ).first()
        
        if existing_user:
            # Update last active timestamp
            existing_user.last_active_at = datetime.utcnow()
            db.commit()
            return existing_user, False
    
    # Create new user
    new_user = User(
        device_identifier=device_identifier,
        onboarding_status=OnboardingStatus.NOT_STARTED,
        created_at=datetime.utcnow(),
        last_active_at=datetime.utcnow()
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    is_new_user = True
    
    logger.info(f"Created new user with ID {new_user.id}")
    return new_user, is_new_user


def validate_user_exists(db: Session, user_id: int) -> User:
    """Validate user exists and return user object."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise_user_not_found(user_id)
    return user


# === API Endpoints ===

@router.post(
    "/initialize",
    response_model=ChatInitializationResponse,
    summary="Initialize chat session",
    description="""
    Called when user first opens the chat page.
    Creates a new user if device_identifier is new, or retrieves existing user.
    Returns user info, recent chat history, and onboarding status.
    """
)
def initialize_chat_session(
    request: ChatInitializationRequest = None,
    db: Session = Depends(get_db)
):
    """
    Initialize or resume a chat session.
    
    - For new users: Creates user, returns empty history, sets onboarding status
    - For returning users: Returns recent history and profile info
    """
    device_id = request.device_identifier if request else None
    
    # Get or create user
    user, is_new_user = get_or_create_user(db, device_id)
    
    # Load recent message history (newest first from DB, then reverse)
    recent_messages = db.query(Message).filter(
        Message.user_id == user.id
    ).order_by(
        Message.id.desc()  # Get newest first
    ).limit(INITIAL_HISTORY_LIMIT).all()
    
    # Reverse to chronological order (oldest first) for display
    recent_messages = list(reversed(recent_messages))
    
    # Check if there's more history beyond what we loaded
    total_message_count = db.query(func.count(Message.id)).filter(
        Message.user_id == user.id
    ).scalar()
    has_more_history = total_message_count > len(recent_messages)
    
    # Convert to response format
    history_response = [message_to_response(msg) for msg in recent_messages]
    
    # Map database enum to schema enum
    onboarding_status_map = {
        OnboardingStatus.NOT_STARTED: OnboardingStatusEnum.NOT_STARTED,
        OnboardingStatus.IN_PROGRESS: OnboardingStatusEnum.IN_PROGRESS,
        OnboardingStatus.COMPLETED: OnboardingStatusEnum.COMPLETED,
    }
    
    logger.info(
        f"Initialized chat for user {user.id}, "
        f"is_new={is_new_user}, "
        f"history_count={len(history_response)}"
    )
    
    return ChatInitializationResponse(
        user_id=user.id,
        onboarding_status=onboarding_status_map.get(
            user.onboarding_status, 
            OnboardingStatusEnum.NOT_STARTED
        ),
        recent_message_history=history_response,
        has_more_history=has_more_history,
        user_display_name=user.display_name
    )


@router.get(
    "/history",
    response_model=ChatHistoryPaginationResponse,
    summary="Get paginated chat history",
    description="""
    Fetch older messages for infinite scroll.
    Uses cursor-based pagination for reliable results.
    Pass cursor_message_id to get messages older than that message.
    """
)
def get_chat_history_paginated(
    user_id: int = Query(..., description="User ID to fetch history for", gt=0),
    cursor_message_id: Optional[int] = Query(
        None, 
        description="Fetch messages older than this message ID. Omit for most recent."
    ),
    page_size: int = Query(
        default=20, 
        description="Number of messages to fetch",
        ge=1,
        le=50
    ),
    db: Session = Depends(get_db)
):
    """
    Paginated history for scroll-to-load-more functionality.
    
    Frontend should:
    1. Call without cursor_message_id to get initial/recent messages
    2. When scrolling up, call with cursor_message_id = oldest loaded message ID
    3. Continue until has_more_messages is False
    """
    # Validate user exists
    user = validate_user_exists(db, user_id)
    
    # Build query for messages
    query = db.query(Message).filter(Message.user_id == user_id)
    
    # Apply cursor filter if provided
    if cursor_message_id is not None:
        query = query.filter(Message.id < cursor_message_id)
    
    # Get messages (newest first, then reverse for display)
    messages = query.order_by(
        Message.id.desc()
    ).limit(page_size + 1).all()  # +1 to check if more exist
    
    # Check if there are more messages
    has_more = len(messages) > page_size
    if has_more:
        messages = messages[:page_size]  # Remove the extra one
    
    # Reverse for chronological order
    messages = list(reversed(messages))
    
    # Get next cursor
    next_cursor = messages[0].id if messages else None
    
    # Get total count for context
    total_count = db.query(func.count(Message.id)).filter(
        Message.user_id == user_id
    ).scalar()
    
    return ChatHistoryPaginationResponse(
        messages=[message_to_response(msg) for msg in messages],
        has_more_messages=has_more,
        next_cursor_message_id=next_cursor,
        total_message_count=total_count
    )


@router.post(
    "/message",
    response_model=SendMessageResponse,
    summary="Send a message and get AI response",
    description="""
    Send a user message and receive AI response.
    The agent uses RAG for medical protocols and long-term user memory.
    """
)
async def send_chat_message(
    request: SendMessageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Main chat endpoint:
    1. Validates input
    2. Saves user message
    3. Fetches context (history, memories, protocols)
    4. Runs AI agent
    5. Saves AI response
    6. Returns both messages
    """
    # Validate user exists
    user = validate_user_exists(db, request.user_id)
    
    # Additional validation
    if len(request.message_content) > MAX_MESSAGE_LENGTH:
        raise_validation_error(
            "message_content",
            f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters"
        )
    
    # 1. Save user message
    user_message = Message(
        user_id=request.user_id,
        role=MessageRole.USER,
        content=request.message_content,
        created_at=datetime.utcnow()
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)
    
    logger.info(f"Saved user message {user_message.id} for user {request.user_id}")
    
    # 2. Fetch recent conversation history for context
    # We load slightly more than needed for agent context
    history_messages = db.query(Message).filter(
        Message.user_id == request.user_id
    ).order_by(
        Message.id.asc()
    ).limit(30).all()  # Recent history for context
    
    # Format messages for agent
    agent_messages = [
        {"role": msg.role.value, "content": msg.content}
        for msg in history_messages
    ]
    
    # 3. Run agent
    try:
        agent_result = run_health_agent(
            db_session=db,
            user_id=request.user_id,
            messages=agent_messages
        )
        ai_response_content = agent_result["final_response"]
        extracted_facts = agent_result.get("extracted_facts", [])
        
    except Exception as e:
        logger.error(f"Agent error for user {request.user_id}: {e}")
        # Graceful fallback
        ai_response_content = (
            "I'm having a little trouble processing that right now. "
            "Could you try rephrasing or let me know how I can help? 🙏"
        )
        extracted_facts = []
    
    # 4. Save AI response
    ai_message = Message(
        user_id=request.user_id,
        role=MessageRole.ASSISTANT,
        content=ai_response_content,
        created_at=datetime.utcnow()
    )
    db.add(ai_message)
    db.commit()
    db.refresh(ai_message)
    
    logger.info(f"Saved AI response {ai_message.id} for user {request.user_id}")
    
    # Update user's last active time
    user.last_active_at = datetime.utcnow()
    db.commit()
    
    # 5. Build response
    extracted_insights = [f["fact_text"] for f in extracted_facts] if extracted_facts else None
    
    return SendMessageResponse(
        user_message=message_to_response(user_message),
        assistant_reply=message_to_response(ai_message),
        extracted_health_insights=extracted_insights
    )


@router.post(
    "/typing",
    response_model=TypingIndicatorResponse,
    summary="Update typing indicator status",
    description="""
    Update user's typing status for real-time chat experience.
    Frontend should call this when user starts/stops typing.
    Status is considered stale after 5 seconds.
    """
)
def update_typing_indicator(
    request: TypingIndicatorRequest,
    db: Session = Depends(get_db)
):
    """
    Update typing indicator for real-time UX.
    
    Frontend should:
    - Send 'started' when user begins typing
    - Send 'stopped' when user stops or sends message
    - Polling endpoint can check if assistant is "typing"
    """
    # Validate user exists
    validate_user_exists(db, request.user_id)
    
    is_typing = request.typing_status == TypingStatus.STARTED
    
    # Upsert typing indicator
    indicator = db.query(TypingIndicator).filter(
        TypingIndicator.user_id == request.user_id
    ).first()
    
    if indicator:
        indicator.is_typing = is_typing
        indicator.last_updated_at = datetime.utcnow()
    else:
        indicator = TypingIndicator(
            user_id=request.user_id,
            is_typing=is_typing,
            last_updated_at=datetime.utcnow()
        )
        db.add(indicator)
    
    db.commit()
    
    return TypingIndicatorResponse(
        acknowledged=True,
        timestamp=datetime.utcnow()
    )


@router.get(
    "/typing/{user_id}",
    response_model=TypingIndicatorResponse,
    summary="Check typing indicator status",
    description="Check if the AI is currently 'typing' a response for a user."
)
def get_typing_indicator(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Check current typing status.
    Returns False if indicator is stale (>5 seconds old).
    """
    validate_user_exists(db, user_id)
    
    indicator = db.query(TypingIndicator).filter(
        TypingIndicator.user_id == user_id
    ).first()
    
    if not indicator:
        return TypingIndicatorResponse(
            acknowledged=False,
            timestamp=datetime.utcnow()
        )
    
    # Check if stale
    is_stale = (
        datetime.utcnow() - indicator.last_updated_at
    ).total_seconds() > TYPING_INDICATOR_TTL_SECONDS
    
    return TypingIndicatorResponse(
        acknowledged=indicator.is_typing and not is_stale,
        timestamp=indicator.last_updated_at
    )


@router.delete(
    "/history/{user_id}",
    summary="Clear chat history",
    description="Clear all chat history for a user. Use with caution."
)
def clear_chat_history(
    user_id: int,
    confirm: bool = Query(
        False, 
        description="Must be true to confirm deletion"
    ),
    db: Session = Depends(get_db)
):
    """
    Clear all messages for a user.
    Requires confirm=true as safety check.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "CONFIRMATION_REQUIRED",
                "error_message": "Set confirm=true to delete history"
            }
        )
    
    user = validate_user_exists(db, user_id)
    
    # Delete messages
    deleted_count = db.query(Message).filter(
        Message.user_id == user_id
    ).delete()
    
    # Also clear summaries
    db.query(ConversationSummary).filter(
        ConversationSummary.user_id == user_id
    ).delete()
    
    db.commit()
    
    logger.info(f"Cleared {deleted_count} messages for user {user_id}")
    
    return {
        "success": True,
        "deleted_message_count": deleted_count,
        "user_id": user_id
    }


@router.get(
    "/user/{user_id}/profile",
    summary="Get user profile and memory",
    description="Get user's profile information and extracted health memories."
)
def get_user_profile(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Get user profile including extracted health facts.
    Useful for debugging and user transparency.
    """
    user = validate_user_exists(db, user_id)
    
    # Get user's memories
    memories = db.query(UserMemoryFact).filter(
        UserMemoryFact.user_id == user_id,
        UserMemoryFact.is_active == True
    ).order_by(
        UserMemoryFact.created_at.desc()
    ).limit(50).all()
    
    return {
        "user_id": user.id,
        "display_name": user.display_name,
        "onboarding_status": user.onboarding_status.value,
        "health_profile_summary": user.health_profile_summary,
        "created_at": user.created_at,
        "last_active_at": user.last_active_at,
        "memory_facts": [
            {
                "id": m.id,
                "fact": m.fact_content,
                "category": m.category.value if m.category else None,
                "extracted_at": m.created_at
            }
            for m in memories
        ]
    }


# === Health Check ===

@router.get(
    "/health",
    summary="Health check endpoint",
    description="Check if the chat service is healthy."
)
def health_check(db: Session = Depends(get_db)):
    """Simple health check that verifies DB connection."""
    try:
        # Quick DB query to verify connection
        db.execute("SELECT 1")
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "service": "chat-api"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "error": str(e)
            }
        )
