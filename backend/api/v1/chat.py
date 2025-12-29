"""
Chat API endpoints for the Health Agent.
Handles initialization, messaging, history pagination, and onboarding.
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
from datetime import datetime
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
    OnboardingSubmissionRequest,
    OnboardingSubmissionResponse,
    ErrorResponse,
    OnboardingStatus as OnboardingStatusEnum,
    MessageRole as MessageRoleEnum,
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
MAX_CONTEXT_MESSAGES = 4  # Last 4 messages for agent context

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
        Message.id.desc()
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


@router.post(
    "/onboarding",
    response_model=OnboardingSubmissionResponse,
    summary="Complete user onboarding",
    description="""
    Submit user profile information during onboarding.
    Collects display name, age, and biological sex.
    """
)
def complete_onboarding(
    request: OnboardingSubmissionRequest,
    db: Session = Depends(get_db)
):
    """
    Complete user onboarding by saving profile information.
    """
    user = validate_user_exists(db, request.user_id)
    
    # Update user profile
    user.display_name = request.display_name
    user.age_years = request.age_years
    user.biological_sex = request.biological_sex
    user.onboarding_status = OnboardingStatus.COMPLETED
    
    db.commit()
    db.refresh(user)
    
    logger.info(f"Completed onboarding for user {user.id}: {user.display_name}")
    
    return OnboardingSubmissionResponse(
        success=True,
        user_id=user.id,
        message=f"Welcome {user.display_name}! Your profile has been saved."
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
    ).limit(page_size + 1).all()
    
    # Check if there are more messages
    has_more = len(messages) > page_size
    if has_more:
        messages = messages[:page_size]
    
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
    The agent uses RAG for medical protocols, rolling summary, and last 4 messages for context.
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
    3. Fetches last 4 messages for agent context
    4. Runs AI agent (uses rolling summary internally)
    5. Saves AI response
    6. Returns both messages
    """
    # Validate user exists
    user = validate_user_exists(db, request.user_id)
    
    # Check onboarding status
    if user.onboarding_status != OnboardingStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ONBOARDING_REQUIRED",
                "error_message": "Please complete onboarding before chatting",
                "details": {"onboarding_status": user.onboarding_status.value}
            }
        )
    
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
        created_at=datetime.utcnow(),
        is_summarized=False
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)
    
    logger.info(f"Saved user message {user_message.id} for user {request.user_id}")
    
    # 2. Fetch last 4 messages for agent context (rolling summary handles the rest)
    history_messages = db.query(Message).filter(
        Message.user_id == request.user_id
    ).order_by(
        Message.id.desc()
    ).limit(MAX_CONTEXT_MESSAGES).all()
    
    # Reverse to chronological order
    history_messages = list(reversed(history_messages))
    
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
        created_at=datetime.utcnow(),
        is_summarized=False
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
