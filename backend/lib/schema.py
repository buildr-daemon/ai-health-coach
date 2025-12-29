"""
Pydantic schemas for API request/response validation.
All models use verbose, descriptive field names for clarity.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


# === Enums ===

class MessageRole(str, Enum):
    """Role of the message sender in the conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class OnboardingStatus(str, Enum):
    """Status of user onboarding process."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TypingStatus(str, Enum):
    """Typing indicator status."""
    STARTED = "started"
    STOPPED = "stopped"


# === Chat Message Models ===

class ChatMessageResponse(BaseModel):
    """
    Represents a single chat message in API responses.
    Contains full metadata for frontend rendering.
    """
    message_id: int = Field(..., description="Unique identifier for the message")
    role: MessageRole = Field(..., description="Who sent this message (user/assistant)")
    content: str = Field(..., description="The actual message text content")
    created_at: datetime = Field(..., description="When the message was created")
    
    class Config:
        from_attributes = True


class ChatMessageForAgent(BaseModel):
    """
    Simplified message format passed to the LLM agent.
    Excludes metadata not needed for context.
    """
    role: MessageRole
    content: str


# === Chat Initialization ===

class ChatInitializationRequest(BaseModel):
    """
    Request to initialize a chat session.
    Can optionally include a device ID for returning users.
    """
    device_identifier: Optional[str] = Field(
        None, 
        description="Unique device ID for identifying returning users",
        max_length=255
    )


class ChatInitializationResponse(BaseModel):
    """
    Response when initializing a new chat session.
    Contains user info, recent history, and onboarding status.
    """
    user_id: int = Field(..., description="Assigned user ID for this session")
    onboarding_status: OnboardingStatus = Field(
        ..., 
        description="Whether user has completed initial onboarding"
    )
    recent_message_history: List[ChatMessageResponse] = Field(
        default_factory=list,
        description="Most recent messages (newest last) for initial display"
    )
    has_more_history: bool = Field(
        False, 
        description="Whether older messages exist for pagination"
    )
    user_display_name: Optional[str] = Field(
        None, 
        description="User's name if known from onboarding"
    )


# === Onboarding ===

class OnboardingSubmissionRequest(BaseModel):
    """
    Request to submit user profile during onboarding.
    Collects essential user information for personalized health guidance.
    """
    user_id: int = Field(..., description="ID of the user to onboard", gt=0)
    display_name: str = Field(
        ..., 
        description="User's preferred name",
        min_length=1,
        max_length=100
    )
    age_years: int = Field(
        ..., 
        description="User's age in years",
        ge=1,
        le=120
    )
    biological_sex: Literal["male", "female", "other"] = Field(
        ..., 
        description="Biological sex for medical context"
    )
    
    @field_validator('display_name')
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        """Ensure name is not just whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Display name cannot be empty")
        return stripped


class OnboardingSubmissionResponse(BaseModel):
    """
    Response after completing onboarding.
    """
    success: bool = Field(..., description="Whether onboarding was successful")
    user_id: int = Field(..., description="The user's ID")
    message: str = Field(..., description="Success or error message")


# === Message Sending ===

class SendMessageRequest(BaseModel):
    """
    Request to send a new message from the user.
    Includes validation for message content.
    """
    user_id: int = Field(..., description="ID of the user sending the message", gt=0)
    message_content: str = Field(
        ..., 
        description="The message text to send",
        min_length=1,
        max_length=4000
    )
    
    @field_validator('message_content')
    @classmethod
    def validate_message_not_empty(cls, value: str) -> str:
        """Ensure message is not just whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or whitespace only")
        return stripped


class SendMessageResponse(BaseModel):
    """
    Response after sending a message.
    Contains both the saved user message and AI reply.
    """
    user_message: ChatMessageResponse = Field(
        ..., 
        description="The user's message as saved"
    )
    assistant_reply: ChatMessageResponse = Field(
        ..., 
        description="The AI assistant's response"
    )
    extracted_health_insights: Optional[List[str]] = Field(
        None,
        description="Any health facts extracted and saved to long-term memory"
    )


# === Chat History Pagination ===

class ChatHistoryPaginationRequest(BaseModel):
    """
    Request for paginated chat history.
    Uses cursor-based pagination for reliable scrolling.
    """
    user_id: int = Field(..., description="User whose history to fetch", gt=0)
    cursor_message_id: Optional[int] = Field(
        None,
        description="ID of the oldest message currently loaded. Fetch messages older than this."
    )
    page_size: int = Field(
        default=20,
        description="Number of messages to fetch",
        ge=1,
        le=50
    )


class ChatHistoryPaginationResponse(BaseModel):
    """
    Paginated chat history response.
    Includes cursor info for subsequent requests.
    """
    messages: List[ChatMessageResponse] = Field(
        default_factory=list,
        description="Messages in chronological order (oldest first)"
    )
    has_more_messages: bool = Field(
        ..., 
        description="Whether more older messages exist"
    )
    next_cursor_message_id: Optional[int] = Field(
        None,
        description="Use this as cursor_message_id for next page"
    )
    total_message_count: int = Field(
        ..., 
        description="Total messages in conversation"
    )


# === Typing Indicator ===

class TypingIndicatorRequest(BaseModel):
    """Request to update typing status."""
    user_id: int = Field(..., description="User whose typing status to update", gt=0)
    typing_status: TypingStatus = Field(..., description="Current typing state")


class TypingIndicatorResponse(BaseModel):
    """Response indicating typing indicator was processed."""
    acknowledged: bool = Field(True, description="Whether the update was processed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Server timestamp of the update"
    )


# === Agent Internal Schemas ===

class RetrievedProtocol(BaseModel):
    """A medical protocol retrieved via RAG."""
    protocol_id: int
    topic: str
    content: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class UserMemoryFact(BaseModel):
    """A fact stored in user's long-term memory."""
    fact_id: int
    fact_content: str
    extracted_at: datetime
    category: Optional[str] = None


class AgentContextBundle(BaseModel):
    """
    Complete context bundle passed to the LLM for response generation.
    Aggregates all relevant information sources.
    """
    user_id: int
    user_display_name: Optional[str]
    user_age: Optional[int]
    user_biological_sex: Optional[str]
    conversation_messages: List[ChatMessageForAgent]
    rolling_summary: Optional[str]
    retrieved_protocols: List[RetrievedProtocol]
    user_memory_facts: List[UserMemoryFact]


# === Health Data Extraction ===

class ExtractedHealthFact(BaseModel):
    """A health fact extracted from conversation."""
    fact_text: str = Field(..., description="The extracted fact statement")
    category: str = Field(..., description="Category: symptom, lifestyle, medical_history, preference")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")


class HealthFactExtractionResult(BaseModel):
    """Result of extracting health facts from a conversation."""
    extracted_facts: List[ExtractedHealthFact]
    should_update_profile: bool = Field(
        False, 
        description="Whether user profile summary should be regenerated"
    )


# === Error Responses ===

class ErrorResponse(BaseModel):
    """Standard error response format."""
    error_code: str = Field(..., description="Machine-readable error code")
    error_message: str = Field(..., description="Human-readable error description")
    details: Optional[dict] = Field(None, description="Additional error context")


class ValidationErrorDetail(BaseModel):
    """Details about a validation error."""
    field_name: str
    error_message: str
    invalid_value: Optional[str] = None
