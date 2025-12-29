"""
SQLAlchemy ORM models for the Health Agent application.
All models use verbose, descriptive column names for clarity.
"""
from sqlalchemy import (
    Column, 
    Integer, 
    String, 
    Text, 
    DateTime, 
    ForeignKey, 
    Boolean,
    Float,
    Enum as SQLEnum,
    Index,
    event
)
from sqlalchemy.orm import (
    relationship, 
    declarative_base,
)
from datetime import datetime
import enum

Base = declarative_base()


# === Enums ===

class OnboardingStatus(enum.Enum):
    """Status of user onboarding process."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class MessageRole(enum.Enum):
    """Role of the message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MemoryCategory(enum.Enum):
    """Category of extracted user memory facts."""
    SYMPTOM = "symptom"
    LIFESTYLE = "lifestyle"
    MEDICAL_HISTORY = "medical_history"
    PREFERENCE = "preference"
    DEMOGRAPHIC = "demographic"
    MEDICATION = "medication"
    ALLERGY = "allergy"


class ProtocolCategory(enum.Enum):
    """Category of medical protocols."""
    SYMPTOM_MANAGEMENT = "symptom_management"
    EMERGENCY_GUIDANCE = "emergency_guidance"
    LIFESTYLE_ADVICE = "lifestyle_advice"
    MEDICATION_INFO = "medication_info"
    POLICY = "policy"  # e.g., refund policies


# === Models ===

class User(Base):
    """
    Represents a user of the health agent.
    Stores profile information gathered during onboarding and conversations.
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Device identification for returning users
    device_identifier = Column(
        String(255), 
        nullable=True, 
        unique=True, 
        index=True,
        comment="Unique device ID for identifying returning users"
    )
    
    # Onboarding status
    onboarding_status = Column(
        SQLEnum(OnboardingStatus),
        default=OnboardingStatus.NOT_STARTED,
        nullable=False,
        comment="Whether user has completed initial onboarding"
    )
    
    # Profile information (gathered during onboarding)
    display_name = Column(
        String(100), 
        nullable=True,
        comment="User's preferred name for personalized responses"
    )
    age_years = Column(
        Integer, 
        nullable=True,
        comment="User's age in years"
    )
    biological_sex = Column(
        String(20), 
        nullable=True,
        comment="Biological sex for medical context"
    )
    
    # Computed health profile (regenerated periodically from memories)
    health_profile_summary = Column(
        Text, 
        nullable=True,
        comment="LLM-generated summary of user's health profile from all memories"
    )
    health_profile_updated_at = Column(
        DateTime, 
        nullable=True,
        comment="When the health profile summary was last regenerated"
    )
    
    # Timestamps
    created_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        nullable=False,
        comment="When the user record was created"
    )
    last_active_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow,
        comment="Last time user interacted with the agent"
    )

    # Relationships
    messages = relationship(
        "Message", 
        back_populates="user", 
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    memory_facts = relationship(
        "UserMemoryFact", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, name={self.display_name}, onboarding={self.onboarding_status})>"


class Message(Base):
    """
    Represents a single message in a conversation.
    Stores both user messages and assistant responses.
    """
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="The user this message belongs to"
    )
    
    role = Column(
        SQLEnum(MessageRole),
        nullable=False,
        comment="Who sent this message (user/assistant/system)"
    )
    content = Column(
        Text, 
        nullable=False,
        comment="The actual message text content"
    )
    
    # Token tracking for context management
    token_count = Column(
        Integer, 
        nullable=True,
        comment="Estimated token count for context window management"
    )
    
    # For summarization: track if message has been summarized into conversation summary
    is_summarized = Column(
        Boolean, 
        default=False,
        nullable=False,
        comment="Whether this message has been condensed into a summary"
    )
    
    created_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        nullable=False,
        index=True,
        comment="When the message was created"
    )
    
    # Relationship
    user = relationship("User", back_populates="messages")
    
    # Index for efficient pagination queries
    __table_args__ = (
        Index('idx_user_messages_pagination', 'user_id', 'id', 'created_at'),
    )
    
    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Message(id={self.id}, role={self.role}, preview='{preview}')>"


class UserMemoryFact(Base):
    """
    Long-term memory facts extracted from conversations.
    These persist across conversations to provide personalized context.
    """
    __tablename__ = "user_memory_facts"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="The user this memory belongs to"
    )
    
    fact_content = Column(
        Text, 
        nullable=False,
        comment="The extracted fact statement (e.g., 'User has diabetes')"
    )
    category = Column(
        SQLEnum(MemoryCategory),
        nullable=False,
        comment="Category of this fact for retrieval filtering"
    )
    
    # Source tracking
    source_message_id = Column(
        Integer, 
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        comment="The message this fact was extracted from"
    )
    extraction_confidence = Column(
        Float, 
        nullable=True,
        comment="LLM confidence score for this extraction (0.0 to 1.0)"
    )
    
    # Validity tracking
    is_active = Column(
        Boolean, 
        default=True, 
        nullable=False,
        comment="Whether this fact is still considered valid/current"
    )
    superseded_by_id = Column(
        Integer, 
        ForeignKey("user_memory_facts.id"),
        nullable=True,
        comment="If this fact was updated, reference to the newer fact"
    )
    
    created_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        nullable=False,
        comment="When this fact was extracted"
    )
    
    # Relationships
    user = relationship("User", back_populates="memory_facts")
    
    def __repr__(self):
        preview = self.fact_content[:50] + "..." if len(self.fact_content) > 50 else self.fact_content
        return f"<UserMemoryFact(id={self.id}, category={self.category}, fact='{preview}')>"


class ConversationSummary(Base):
    """
    Stores compressed summaries of old messages.
    Used for context overflow handling - old messages are summarized
    and removed from active context.
    """
    __tablename__ = "conversation_summaries"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="The user this summary belongs to"
    )
    
    summary_content = Column(
        Text, 
        nullable=False,
        comment="Compressed summary of older conversation messages"
    )
    
    # Track what messages this summary covers
    covers_messages_from_id = Column(
        Integer, 
        nullable=False,
        comment="Oldest message ID included in this summary"
    )
    covers_messages_to_id = Column(
        Integer, 
        nullable=False,
        comment="Newest message ID included in this summary"
    )
    message_count_summarized = Column(
        Integer, 
        nullable=False,
        comment="Number of messages condensed into this summary"
    )
    
    created_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        nullable=False,
        comment="When this summary was generated"
    )
    
    def __repr__(self):
        return f"<ConversationSummary(id={self.id}, user={self.user_id}, msgs={self.message_count_summarized})>"


class MedicalProtocol(Base):
    """
    Static knowledge base for medical guidance and policies.
    Used for RAG - matched against user queries in real-time.
    """
    __tablename__ = "medical_protocols"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Identification
    protocol_code = Column(
        String(50), 
        unique=True, 
        nullable=False,
        comment="Unique code for this protocol (e.g., 'FEVER_MGMT_001')"
    )
    title = Column(
        String(200), 
        nullable=False,
        comment="Human-readable title of the protocol"
    )
    
    category = Column(
        SQLEnum(ProtocolCategory),
        nullable=False,
        index=True,
        comment="Category for filtering"
    )
    
    # Content
    content = Column(
        Text, 
        nullable=False,
        comment="The actual protocol/guideline content"
    )
    
    # Keywords for simple matching (before vector search)
    keywords = Column(
        Text, 
        nullable=True,
        comment="Comma-separated keywords for basic matching"
    )
    
    # Metadata
    severity_level = Column(
        Integer, 
        default=1,
        comment="1=general advice, 2=caution needed, 3=seek doctor, 4=emergency"
    )
    requires_professional_followup = Column(
        Boolean, 
        default=False,
        comment="Whether this condition typically needs doctor consultation"
    )
    
    # Status
    is_active = Column(
        Boolean, 
        default=True, 
        nullable=False,
        comment="Whether this protocol is currently active"
    )
    
    created_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        nullable=False
    )
    updated_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )
    
    def __repr__(self):
        return f"<MedicalProtocol(code={self.protocol_code}, title='{self.title}')>"


class TypingIndicator(Base):
    """
    Tracks typing status for real-time chat experience.
    Uses short TTL - records are considered stale after a few seconds.
    """
    __tablename__ = "typing_indicators"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        unique=True,  # One record per user
        comment="The user whose typing status this tracks"
    )
    
    is_typing = Column(
        Boolean, 
        default=False,
        nullable=False,
        comment="Whether the user is currently typing"
    )
    
    last_updated_at = Column(
        DateTime, 
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        comment="When typing status was last updated (stale after 5s)"
    )
    
    def __repr__(self):
        return f"<TypingIndicator(user={self.user_id}, typing={self.is_typing})>"


# === Helper function to create all tables ===

def create_all_tables(engine):
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def seed_medical_protocols(db_session):
    """
    Seed the database with common medical protocols.
    Should be called once during initial setup.
    """
    protocols = [
        # Fever Management
        MedicalProtocol(
            protocol_code="FEVER_MGMT_001",
            title="Fever Management Guidelines",
            category=ProtocolCategory.SYMPTOM_MANAGEMENT,
            content="""
            For fever management:
            - Temperature 99-100.4°F (37.2-38°C): Low-grade fever. Rest, stay hydrated.
            - Temperature 100.4-103°F (38-39.4°C): Moderate fever. Take acetaminophen or ibuprofen. Cool compress.
            - Temperature >103°F (39.4°C): High fever. SEEK MEDICAL ATTENTION if it doesn't respond to medication.
            - Duration >3 days: CONSULT A DOCTOR regardless of temperature.
            
            Red flags requiring immediate attention:
            - Fever with stiff neck, severe headache, confusion
            - Fever with difficulty breathing
            - Fever in infants under 3 months
            - Temperature >104°F (40°C) that doesn't respond to medication
            
            General care: Rest, drink plenty of fluids (water, clear broths, electrolyte drinks), 
            dress in light clothing, keep room cool but not cold.
            """,
            keywords="fever,temperature,hot,burning,chills,sweating,thermometer",
            severity_level=2,
            requires_professional_followup=False
        ),
        
        # Headache
        MedicalProtocol(
            protocol_code="HEADACHE_001",
            title="Headache Assessment and Management",
            category=ProtocolCategory.SYMPTOM_MANAGEMENT,
            content="""
            For headache assessment:
            - Tension headache: Dull, aching pain, tightness around forehead. Usually responds to rest and OTC pain relievers.
            - Migraine: Throbbing pain, often one-sided, may have nausea, light sensitivity. Dark quiet room helps.
            - Cluster headache: Severe pain around one eye, often with eye watering.
            
            Self-care steps:
            1. Rest in a dark, quiet room
            2. Apply cold or warm compress
            3. Stay hydrated
            4. OTC pain relievers (acetaminophen, ibuprofen) if not contraindicated
            5. Avoid triggers (bright lights, loud sounds, strong smells)
            
            SEEK IMMEDIATE CARE if headache is:
            - Sudden and severe ("worst headache of life")
            - Accompanied by fever, stiff neck, confusion, seizures
            - Following head injury
            - Getting progressively worse over days
            - Accompanied by vision changes, weakness, or speech problems
            """,
            keywords="headache,head,pain,migraine,throbbing,tension",
            severity_level=2,
            requires_professional_followup=False
        ),
        
        # Stomach/Digestive Issues
        MedicalProtocol(
            protocol_code="STOMACH_001",
            title="Stomach Ache and Digestive Issues",
            category=ProtocolCategory.SYMPTOM_MANAGEMENT,
            content="""
            For stomach discomfort:
            
            MILD SYMPTOMS (self-care):
            - Indigestion/bloating: Avoid fatty foods, eat smaller meals, try ginger or peppermint tea
            - Mild nausea: Clear fluids, bland foods (BRAT: bananas, rice, applesauce, toast)
            - Mild diarrhea: Stay hydrated with electrolytes, avoid dairy and fatty foods
            
            CONCERNING SYMPTOMS (consult doctor within 24-48h):
            - Pain lasting >24 hours
            - Persistent vomiting >24 hours
            - Blood in vomit or stool
            - Unable to keep fluids down
            - Fever with stomach pain
            
            EMERGENCY (seek immediate care):
            - Severe abdominal pain (especially lower right - could indicate appendicitis)
            - Signs of dehydration (dark urine, dizziness, rapid heartbeat)
            - Vomiting blood or "coffee grounds" material
            - Black, tarry stools
            - Severe pain after injury to abdomen
            - Pregnant with abdominal pain or bleeding
            """,
            keywords="stomach,belly,abdomen,nausea,vomit,diarrhea,indigestion,bloating,cramps,pain",
            severity_level=2,
            requires_professional_followup=False
        ),
        
        # Cold/Flu
        MedicalProtocol(
            protocol_code="COLD_FLU_001",
            title="Cold and Flu Management",
            category=ProtocolCategory.SYMPTOM_MANAGEMENT,
            content="""
            Cold vs Flu distinction:
            - Cold: Gradual onset, mild symptoms, mainly runny nose and sore throat
            - Flu: Sudden onset, fever, body aches, fatigue, more severe symptoms
            
            Self-care for mild symptoms:
            1. REST is crucial - your body needs energy to fight infection
            2. HYDRATION - water, warm broths, herbal teas, electrolyte drinks
            3. For congestion: steam inhalation, saline nasal rinse
            4. For sore throat: warm salt water gargle, honey (adults only)
            5. OTC medications: decongestants, pain relievers, cough suppressants as needed
            
            Duration: Cold typically 7-10 days, Flu 1-2 weeks
            
            SEEK MEDICAL ATTENTION if:
            - Difficulty breathing or shortness of breath
            - Persistent chest pain or pressure
            - Confusion or altered mental state
            - Severe or persistent vomiting
            - Symptoms improve then return worse (secondary infection)
            - High-risk groups: elderly, pregnant, immunocompromised, chronic conditions
            """,
            keywords="cold,flu,cough,sneeze,runny nose,congestion,sore throat,body ache,fatigue",
            severity_level=1,
            requires_professional_followup=False
        ),
        
        # Back Pain
        MedicalProtocol(
            protocol_code="BACK_PAIN_001",
            title="Back Pain Assessment",
            category=ProtocolCategory.SYMPTOM_MANAGEMENT,
            content="""
            For back pain management:
            
            Most back pain is muscular and resolves with:
            - Rest (but avoid prolonged bed rest - gentle movement helps)
            - Ice for first 48-72 hours, then heat
            - OTC pain relievers (ibuprofen, acetaminophen)
            - Gentle stretching and movement
            - Good posture and ergonomics
            
            Prevention:
            - Regular exercise, especially core strengthening
            - Proper lifting technique (bend knees, not back)
            - Ergonomic workspace setup
            - Maintain healthy weight
            
            SEEK IMMEDIATE CARE (Red flags):
            - Back pain with loss of bladder/bowel control
            - Numbness in groin or inner thighs
            - Weakness in legs or difficulty walking
            - Pain after significant trauma/fall
            - Pain with fever and no other explanation
            - Unexplained weight loss with back pain
            
            See doctor if pain persists >2 weeks despite self-care.
            """,
            keywords="back,pain,spine,lower back,upper back,muscle,posture,lifting",
            severity_level=2,
            requires_professional_followup=False
        ),
        
        # Allergies
        MedicalProtocol(
            protocol_code="ALLERGY_001",
            title="Allergy Symptoms Management",
            category=ProtocolCategory.SYMPTOM_MANAGEMENT,
            content="""
            For allergy symptoms:
            
            MILD allergic reactions:
            - Sneezing, runny/stuffy nose, itchy eyes: Antihistamines (cetirizine, loratadine)
            - Itchy skin, mild hives: Antihistamines, cool compress, avoid scratching
            - Identify and avoid triggers when possible
            
            Management tips:
            - Keep windows closed during high pollen days
            - Shower after outdoor activities
            - Use air purifiers indoors
            - Nasal saline rinse for congestion
            
            EMERGENCY - Anaphylaxis signs (CALL EMERGENCY SERVICES):
            - Difficulty breathing or swallowing
            - Swelling of throat, tongue, or lips
            - Rapid heartbeat
            - Dizziness or fainting
            - Severe hives covering large body area
            - Known severe allergy after exposure
            
            If you have an EpiPen, use it immediately and still call emergency services.
            """,
            keywords="allergy,allergic,hives,itching,sneezing,reaction,swelling,rash",
            severity_level=2,
            requires_professional_followup=False
        ),
        
        # Sleep Issues
        MedicalProtocol(
            protocol_code="SLEEP_001",
            title="Sleep Issues and Insomnia",
            category=ProtocolCategory.LIFESTYLE_ADVICE,
            content="""
            For better sleep (Sleep Hygiene):
            
            1. CONSISTENT SCHEDULE: Same bedtime and wake time daily, even weekends
            2. SLEEP ENVIRONMENT: Dark, cool (65-68°F), quiet, comfortable
            3. WIND-DOWN ROUTINE: 30-60 min before bed without screens
            4. LIMIT STIMULANTS: No caffeine after 2pm, limit alcohol
            5. EXERCISE: Regular activity, but not within 3 hours of bedtime
            6. BED = SLEEP: Don't work, eat, or watch TV in bed
            7. IF CAN'T SLEEP: Get up after 20 min, do calm activity, return when sleepy
            
            Natural aids that may help:
            - Chamomile tea
            - Magnesium supplement
            - Melatonin (short-term, consult doctor)
            - Relaxation techniques, meditation
            
            See a doctor if:
            - Sleep problems persist >3 weeks
            - Excessive daytime sleepiness affecting function
            - Snoring with gasping/stopping breathing (possible sleep apnea)
            - Restless legs or unusual movements during sleep
            """,
            keywords="sleep,insomnia,tired,fatigue,can't sleep,wake up,rest,exhausted",
            severity_level=1,
            requires_professional_followup=False
        ),
        
        # Anxiety/Stress
        MedicalProtocol(
            protocol_code="STRESS_001",
            title="Stress and Anxiety Management",
            category=ProtocolCategory.LIFESTYLE_ADVICE,
            content="""
            For managing stress and anxiety:
            
            IMMEDIATE calming techniques:
            - Deep breathing: 4-7-8 method (inhale 4s, hold 7s, exhale 8s)
            - Grounding: 5-4-3-2-1 (5 things you see, 4 hear, 3 touch, 2 smell, 1 taste)
            - Progressive muscle relaxation
            - Step away from stressor if possible
            
            Long-term management:
            - Regular exercise (proven anxiety reducer)
            - Adequate sleep
            - Limit caffeine and alcohol
            - Maintain social connections
            - Practice mindfulness or meditation
            - Set boundaries, learn to say no
            
            I'm an AI and not a substitute for mental health support.
            
            PLEASE SEEK PROFESSIONAL HELP if:
            - Anxiety is interfering with daily life
            - Panic attacks occurring regularly
            - Physical symptoms (racing heart, sweating) are frequent
            - Feeling hopeless or having thoughts of self-harm
            - Using substances to cope
            
            Crisis resources: National crisis line, therapist, or doctor can help.
            """,
            keywords="stress,anxiety,anxious,worried,panic,nervous,overwhelmed,tension,calm",
            severity_level=1,
            requires_professional_followup=True
        ),
        
        # Skin Issues
        MedicalProtocol(
            protocol_code="SKIN_001",
            title="Common Skin Issues",
            category=ProtocolCategory.SYMPTOM_MANAGEMENT,
            content="""
            For common skin concerns:
            
            RASH (general):
            - Keep area clean and dry
            - Avoid scratching
            - OTC hydrocortisone cream for itching
            - Cool compress for relief
            
            DRY SKIN: Moisturize after bathing, use gentle soaps, humidifier
            
            MINOR BURNS: Cool water (not ice) for 10-20 min, aloe vera, don't pop blisters
            
            INSECT BITES: Clean area, cold compress, antihistamine for itching
            
            ACNE: Gentle cleansing, don't pick, OTC benzoyl peroxide or salicylic acid
            
            SEE A DOCTOR if:
            - Rash is spreading rapidly
            - Rash with fever
            - Signs of infection (increasing redness, warmth, pus, red streaks)
            - Rash after starting new medication
            - Severe blistering
            - Rash that doesn't improve after 1-2 weeks of self-care
            
            EMERGENCY if rash accompanies difficulty breathing or swelling.
            """,
            keywords="skin,rash,itch,hives,burn,dry,acne,bump,red,irritation",
            severity_level=1,
            requires_professional_followup=False
        ),
        
        # Empathy Protocol
        MedicalProtocol(
            protocol_code="EMPATHY_001",
            title="Empathetic Response Guidelines",
            category=ProtocolCategory.POLICY,
            content="""
            Communication guidelines for health conversations:
            
            ALWAYS:
            1. Acknowledge the person's concern first before giving advice
            2. Use warm, conversational language (like WhatsApp, not clinical)
            3. Validate their feelings ("That sounds uncomfortable", "I understand that's worrying")
            4. Ask clarifying questions if symptoms are vague
            5. Give actionable, specific advice
            6. Include appropriate disclaimers for serious symptoms
            
            TONE:
            - Friendly and approachable, like a knowledgeable friend
            - Not condescending or overly clinical
            - Show care and concern
            - Use simple language, avoid jargon
            
            STRUCTURE:
            - Short paragraphs, easy to read on mobile
            - Bullet points for lists
            - Most important info first
            - Clear next steps
            
            NEVER:
            - Diagnose specific conditions
            - Recommend prescription medications
            - Dismiss concerns as "nothing"
            - Give generic responses without personalization
            """,
            keywords="communication,empathy,response,tone,style",
            severity_level=1,
            requires_professional_followup=False
        ),
        
        # Refund/Service Policy
        MedicalProtocol(
            protocol_code="REFUND_001",
            title="Service and Refund Policies",
            category=ProtocolCategory.POLICY,
            content="""
            Service policies:
            
            RESPONSE TIMES:
            - We aim to respond within a few seconds to messages
            - For urgent symptoms, we provide immediate guidance
            - This is not a replacement for emergency services
            
            LIMITATIONS:
            - We provide health information and guidance, not medical diagnosis
            - For emergencies, always call local emergency services
            - This service supplements, not replaces, professional medical care
            
            DATA & PRIVACY:
            - Your health information is stored securely
            - We use your data only to provide personalized health guidance
            - You can request deletion of your data at any time
            
            REFUND POLICY (if applicable):
            - For subscription services, contact support for refund requests
            - Refunds processed within 5-7 business days
            - Pro-rated refunds available for annual subscriptions
            """,
            keywords="refund,policy,support,service,privacy,data,subscription",
            severity_level=1,
            requires_professional_followup=False
        ),
    ]
    
    # Check if protocols already exist
    existing_count = db_session.query(MedicalProtocol).count()
    if existing_count == 0:
        for protocol in protocols:
            db_session.add(protocol)
        db_session.commit()
        return len(protocols)
    return 0
