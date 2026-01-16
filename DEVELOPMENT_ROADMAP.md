# Resume Griller - Development Roadmap

> **Last Updated**: 2026-01-16
> **Current Branch**: heuristic-swirles
> **Codebase Size**: 7,478 lines (Python backend + RAG)
> **Status**: Working prototype with production readiness gaps

---

## 📋 Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Development Directions](#development-directions)
3. [Prioritized Roadmap](#prioritized-roadmap)
4. [Quick Wins](#quick-wins)
5. [Technical Debt](#technical-debt)
6. [Architecture Evolution](#architecture-evolution)

---

## 🎯 Current State Assessment

### ✅ What's Working (Production-Ready)

| Component | Status | Lines | Quality |
|-----------|--------|-------|---------|
| **Grilling Engine** | ✅ Excellent | 973 | 18 gap types, 7D scoring, hybrid mode support |
| **LLM Service** | ✅ Good | 828 | 6 providers (Groq, Gemini, OpenAI, Claude, Custom, Hybrid) |
| **RAG Pipeline** | ✅ Excellent | 1,238 | ChromaDB, semantic chunking, working parser |
| **Interview Agent** | ✅ Good | 596 | API mode orchestration, resume consistency |
| **WebSocket Handler** | ✅ Good | 776 | Real-time communication, voice integration |
| **Voice Services** | ✅ Good | 449 | Deepgram STT, ElevenLabs TTS |
| **Frontend UI** | ✅ Good | ~1,500 | React 19, Next.js 16, video/voice integration |

**Total Working Code**: ~6,360 lines

---

### 🔴 Critical Issues (Must Fix Before Production)

#### 1. **Session Persistence** - CRITICAL
- **Problem**: All sessions stored in-memory, lost on restart
- **Location**: `backend/app/db/session_store.py:235`
- **Impact**: Interviews interrupted by deployment are completely lost
- **Solution Required**: PostgreSQL/Redis backend with recovery mechanism

#### 2. **Test Coverage** - HIGH PRIORITY
- **Current**: <5% (only `tests/test_parser.py` with 47 lines)
- **Missing**:
  - No tests for GrillingEngine (973 lines)
  - No tests for LLM Service (828 lines)
  - No tests for WebSocket handler (776 lines)
  - No integration tests
- **Target**: 70%+ coverage

#### 3. **Duplicate Code** - HIGH PRIORITY
- **Location**:
  - `backend/app/core/resume_parser.py` (175 lines - stub, not used)
  - `scripts/backend/app/core/resume_parser.py` (duplicate stub)
- **Action**: Delete duplicates, use only `rag/resume_parser.py`

#### 4. **Security Gaps** - HIGH PRIORITY
- No authentication/authorization
- No rate limiting on LLM API calls
- No session expiration
- No input validation on uploads

#### 5. **Logging** - MEDIUM PRIORITY
- Using `print()` statements throughout codebase
- No structured logging
- No log levels (DEBUG/INFO/ERROR)

---

### ⚠️ Recent Development Context

**Latest Commit** (2026-01-16):
```
7bafa18 - feat: add langgraph dependencies for multi-agent refactor
```

**Key Insight**: Team is planning a **LangGraph-based multi-agent refactor**:
- Dependencies added: `langgraph>=1.0.6`, `langgraph-checkpoint-sqlite>=3.0.2`
- **Current code does NOT use LangGraph yet** - this is preparatory
- Suggests major architectural change incoming

**Recent Focus Areas**:
- Hybrid model support (Groq preprocessing + Custom model execution)
- Video UI integration
- Enhanced grilling engine with 18 gap types
- HR/Tech mode improvements

---

## 🚀 Development Directions

### Direction 1: Multi-Agent Architecture Refactor ⭐ TOP PRIORITY

**Status**: Dependencies added, not yet implemented
**Effort**: 3-4 weeks
**Impact**: High (solves session persistence, enables advanced features)

#### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph State Machine                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │  Question    │  →   │  Evaluator   │  →   │ FollowUp │ │
│  │  Generator   │      │    Agent     │      │  Agent   │ │
│  │   Agent      │      │              │      │          │ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│         ↓                      ↓                    ↓      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         LangGraph Checkpoints (SQLite)              │  │
│  │         → Persistent state across restarts          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### Agent Breakdown

**QuestionGeneratorAgent**
- Input: Resume, interview mode (HR/Tech/Mixed)
- Output: List of targeted questions
- Reuses: Current `InterviewAgent.generate_questions()`
- Enhancement: Add difficulty adaptation

**EvaluatorAgent**
- Input: Question, answer, resume context
- Output: 7D scores + 18 gap types
- Reuses: Current `GrillingEngine.evaluate_answer()`
- Enhancement: Domain-specific scoring models

**FollowUpAgent**
- Input: Evaluation results, gaps detected
- Output: Targeted follow-up question
- Reuses: Current `GrillingEngine.generate_follow_up()`
- Enhancement: Multi-turn context awareness

**SummaryAgent** (NEW)
- Input: Full interview transcript, all evaluations
- Output: Comprehensive feedback report
- Features:
  - Overall score with percentile ranking
  - Strength/weakness analysis
  - Personalized improvement suggestions

#### Benefits

1. **Solves Session Persistence**: LangGraph checkpoints handle state automatically
2. **Enables Visualization**: Use LangGraph Studio to debug interview flows
3. **Better Error Recovery**: Can resume from any checkpoint
4. **Scalability**: Easy to add new agents (e.g., CodingChallengeAgent)

#### Migration Strategy

**Phase 1**: Checkpoint Infrastructure (Week 1)
```python
# Add checkpoint configuration
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("./data/interview_checkpoints.db")
```

**Phase 2**: Convert InterviewAgent (Week 2)
```python
# Migrate to LangGraph StateGraph
from langgraph.graph import StateGraph

class InterviewState(TypedDict):
    session_id: str
    resume_id: str
    questions: list[str]
    current_question_idx: int
    conversation_history: list[dict]
    evaluations: list[dict]

workflow = StateGraph(InterviewState)
workflow.add_node("generate_questions", question_generator_agent)
workflow.add_node("evaluate_answer", evaluator_agent)
workflow.add_node("generate_follow_up", follow_up_agent)
# ... define edges and conditions
```

**Phase 3**: Integrate with WebSocket (Week 3)
```python
# Update websocket.py to use LangGraph workflow
async def handle_interview_message(websocket, message, session_id):
    # Load state from checkpoint
    state = await checkpointer.get(session_id)

    # Execute workflow step
    result = await workflow.ainvoke(
        {"user_message": message},
        config={"configurable": {"thread_id": session_id}}
    )

    # State automatically saved to checkpoint
    await websocket.send_json(result)
```

**Phase 4**: Testing & Rollout (Week 4)
- Integration tests with checkpoints
- Load testing for concurrent interviews
- Gradual rollout with feature flag

---

### Direction 2: Production Readiness 🔴 CRITICAL

**Status**: Major gaps identified
**Effort**: 2-3 weeks
**Impact**: Critical (blocking production deployment)

#### 2.1 Persistent Session Storage

**Current State**:
```python
# backend/app/db/session_store.py
class SessionStore:
    def __init__(self):
        self.sessions: Dict[str, InterviewSession] = {}  # In-memory only!
```

**Migration Path**:

**Option A: PostgreSQL** (Recommended for production)
```python
# Use SQLAlchemy with async support
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

class PostgresSessionStore:
    async def save_session(self, session: InterviewSession):
        async with AsyncSession(self.engine) as db:
            db_session = SessionModel(
                id=session.session_id,
                resume_id=session.resume_id,
                state=session.model_dump_json(),
                created_at=session.created_at
            )
            await db.merge(db_session)
            await db.commit()

    async def get_session(self, session_id: str) -> InterviewSession:
        async with AsyncSession(self.engine) as db:
            result = await db.get(SessionModel, session_id)
            return InterviewSession.model_validate_json(result.state)
```

**Option B: Redis** (Recommended for hybrid approach)
```python
# Fast access for active sessions + PostgreSQL for archival
import redis.asyncio as redis

class HybridSessionStore:
    def __init__(self):
        self.redis = redis.from_url("redis://localhost")
        self.postgres = PostgresSessionStore()

    async def save_session(self, session: InterviewSession):
        # Hot storage (Redis) - 24h TTL
        await self.redis.setex(
            f"session:{session.session_id}",
            86400,  # 24 hours
            session.model_dump_json()
        )
        # Cold storage (PostgreSQL) - permanent
        await self.postgres.save_session(session)
```

**Migration Steps**:
1. Create SQLAlchemy models
2. Add Alembic for migrations
3. Implement PostgresSessionStore
4. Add compatibility layer (support both in-memory and Postgres)
5. Update all `SessionStore` usage
6. Add session recovery endpoint

**Files to Modify**:
- `backend/app/db/session_store.py` - Add Postgres backend
- `backend/app/api/routes/session.py` - Update create/get endpoints
- `backend/app/api/routes/websocket.py` - Load from Postgres
- Add `alembic/versions/001_create_sessions_table.py`

---

#### 2.2 Testing Infrastructure

**Target Coverage**: 70%+

**Test Structure**:
```
tests/
├── unit/
│   ├── test_grilling_engine.py      # Test 18 gap types
│   ├── test_interview_agent.py      # Test question generation
│   ├── test_llm_service.py          # Test provider switching
│   └── test_rag_pipeline.py         # Test chunking/retrieval
├── integration/
│   ├── test_websocket_flow.py       # Full interview flow
│   ├── test_voice_services.py       # STT/TTS integration
│   └── test_session_persistence.py  # Save/load sessions
├── fixtures/
│   ├── sample_resumes.py            # Resume fixtures
│   ├── mock_llm_responses.py        # Mock LLM calls
│   └── sample_sessions.py           # Session fixtures
└── conftest.py                      # Pytest configuration
```

**Priority Tests to Write**:

**1. GrillingEngine (High Priority)**
```python
# tests/unit/test_grilling_engine.py
import pytest
from backend.app.core.grilling_engine import GrillingEngine, GapType

@pytest.mark.asyncio
async def test_detect_no_specific_example():
    """Test gap detection: no_specific_example"""
    engine = GrillingEngine(model_type="api")

    question = "Tell me about a challenging project."
    answer = "I worked on challenging projects and solved problems."

    evaluation = await engine.evaluate_answer(
        question=question,
        answer=answer,
        resume_context="Senior Software Engineer at TechCorp"
    )

    assert GapType.NO_SPECIFIC_EXAMPLE in evaluation.gaps
    assert evaluation.scores.specificity < 0.5

@pytest.mark.asyncio
async def test_detect_no_metrics():
    """Test gap detection: no_metrics"""
    answer = "I improved the system performance significantly."
    # Should detect lack of quantifiable metrics

@pytest.mark.asyncio
async def test_forced_first_followup():
    """Verify first answer always gets follow-up"""
    # Even with perfect score, should ask follow-up
```

**2. LLM Service Provider Switching (High Priority)**
```python
# tests/unit/test_llm_service.py
@pytest.mark.asyncio
async def test_provider_switching():
    """Test switching between Groq, Gemini, OpenAI"""
    # Mock all provider APIs
    # Verify correct provider called based on config

@pytest.mark.asyncio
async def test_hybrid_mode_preprocessing():
    """Test Hybrid mode uses Groq for preprocessing"""
    # Verify summary generation uses Groq
    # Verify evaluation uses Custom model
```

**3. WebSocket Flow (Integration)**
```python
# tests/integration/test_websocket_flow.py
@pytest.mark.asyncio
async def test_complete_interview_flow():
    """Test full interview from start to finish"""
    async with websocket_connect(f"ws://localhost:8000/ws/interview/{session_id}") as ws:
        # 1. Receive first question
        msg = await ws.receive_json()
        assert msg["type"] == "question"

        # 2. Send answer with gaps
        await ws.send_json({
            "type": "answer",
            "content": "I worked on microservices."
        })

        # 3. Receive follow-up
        msg = await ws.receive_json()
        assert msg["type"] == "follow_up"
        assert "specific" in msg["content"].lower()

        # ... continue through interview
```

**Implementation Timeline**:
- Week 1: Unit tests for GrillingEngine + LLM Service
- Week 2: Integration tests for WebSocket + Voice
- Week 3: Fixtures + pytest configuration + CI/CD

---

#### 2.3 Security Enhancements

**Authentication System**:
```python
# backend/app/core/auth.py
from jose import jwt
from passlib.context import CryptContext

class AuthService:
    def create_access_token(self, user_id: str) -> str:
        """Generate JWT token"""

    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT"""

# Protect endpoints
@router.post("/sessions")
async def create_session(
    resume_id: str,
    current_user: dict = Depends(get_current_user)  # Add auth
):
    # Only authenticated users can create sessions
```

**Rate Limiting**:
```python
# backend/app/middleware/rate_limit.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Apply to LLM-heavy endpoints
@router.post("/sessions/{session_id}/answer")
@limiter.limit("10/minute")  # Max 10 answers per minute
async def submit_answer(...):
    pass
```

**Input Validation**:
```python
# backend/app/models/schemas.py
from pydantic import validator, Field

class CreateSessionRequest(BaseModel):
    resume_id: str = Field(..., regex=r'^resume_[a-zA-Z0-9]{8,}$')
    mode: InterviewMode  # Enum validation
    num_questions: int = Field(ge=3, le=10)  # 3-10 questions
    max_follow_ups: int = Field(ge=1, le=5)  # 1-5 follow-ups

    @validator('resume_id')
    def validate_resume_exists(cls, v):
        # Check resume exists in database
        if not resume_exists(v):
            raise ValueError("Resume not found")
        return v
```

---

#### 2.4 Structured Logging

**Replace all print() statements**:
```python
# backend/app/core/logging_config.py
import structlog

def configure_logging():
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

# Usage in grilling_engine.py
logger = structlog.get_logger()

# Replace: print(f"[GrillingEngine] Evaluating answer...")
# With:
logger.info("evaluating_answer",
    question_id=question_id,
    answer_length=len(answer),
    model_type=self.model_type
)
```

**Log Levels**:
- DEBUG: Detailed LLM prompts/responses
- INFO: Interview progress, question generation
- WARNING: Fallback evaluations, missing context
- ERROR: LLM API failures, WebSocket errors

---

### Direction 3: Enhanced Grilling Intelligence 🧠

**Status**: Current system is good, can be excellent
**Effort**: 4-6 weeks
**Impact**: High (core product differentiation)

#### 3.1 Domain-Specific Scoring Models

**Problem**: Same scoring criteria for all roles (frontend, backend, ML, etc.)

**Solution**: Role-aware evaluation

```python
# backend/app/core/grilling_engine_v2.py
class DomainSpecificGrillingEngine(GrillingEngine):
    DOMAIN_WEIGHTS = {
        "frontend": {
            "technical_depth": 0.8,      # Less emphasis on architecture
            "clarity": 1.2,              # More emphasis on UX reasoning
            "specificity": 1.0,
        },
        "backend": {
            "technical_depth": 1.3,      # High emphasis on architecture
            "quantification": 1.2,       # Performance metrics critical
            "specificity": 1.1,
        },
        "ml_engineer": {
            "technical_depth": 1.4,      # Highest technical emphasis
            "quantification": 1.5,       # Metrics (accuracy, precision) critical
            "informativeness": 1.2,      # Model selection reasoning
        }
    }

    def __init__(self, model_type: str, domain: str = "general"):
        super().__init__(model_type)
        self.domain = domain
        self.weights = self.DOMAIN_WEIGHTS.get(domain, {})

    def calculate_weighted_score(self, scores: DetailedScores) -> float:
        """Apply domain-specific weights to scores"""
        weighted = {
            "relevancy": scores.relevancy * self.weights.get("relevancy", 1.0),
            "clarity": scores.clarity * self.weights.get("clarity", 1.0),
            "technical_depth": scores.depth * self.weights.get("technical_depth", 1.0),
            # ...
        }
        return sum(weighted.values()) / sum(self.weights.values())
```

**Additional Domain-Specific Gap Types**:
```python
# For ML roles
GapType.NO_MODEL_COMPARISON = "no_model_comparison"  # Didn't compare model alternatives
GapType.NO_EVALUATION_METRICS = "no_evaluation_metrics"  # No precision/recall/F1

# For Leadership roles
GapType.NO_STAKEHOLDER_MANAGEMENT = "no_stakeholder_management"
GapType.NO_DECISION_RATIONALE = "no_decision_rationale"

# For Frontend roles
GapType.NO_ACCESSIBILITY_CONSIDERATION = "no_accessibility"
GapType.NO_PERFORMANCE_OPTIMIZATION = "no_performance_metrics"
```

---

#### 3.2 Adaptive Difficulty System

**Current**: All candidates get same difficulty questions

**Improvement**: Adjust based on performance

```python
# backend/app/core/adaptive_interviewer.py
class AdaptiveInterviewer:
    DIFFICULTY_THRESHOLDS = {
        "easy": 0.0,
        "medium": 0.65,
        "hard": 0.85
    }

    def determine_next_difficulty(self, evaluations: list[AnswerEvaluation]) -> str:
        """Adjust difficulty based on recent performance"""
        recent_scores = [e.overall_score for e in evaluations[-3:]]  # Last 3 answers
        avg_score = sum(recent_scores) / len(recent_scores)

        if avg_score >= self.DIFFICULTY_THRESHOLDS["hard"]:
            return "hard"  # Challenge them with architecture/scale questions
        elif avg_score >= self.DIFFICULTY_THRESHOLDS["medium"]:
            return "medium"  # Standard depth
        else:
            return "easy"  # Back to fundamentals

    async def generate_adaptive_question(
        self,
        resume_context: str,
        difficulty: str,
        focus_area: str
    ) -> str:
        """Generate question matching difficulty level"""

        prompts = {
            "easy": "Ask a basic question about {focus_area} fundamentals",
            "medium": "Ask about a real-world scenario involving {focus_area}",
            "hard": "Ask about trade-offs when scaling {focus_area} to millions of users"
        }

        # Use LLM with difficulty-specific prompt
```

**Example Progression**:
```
Question 1 (Medium): "Tell me about your microservices experience."
Answer Score: 0.45 (low) → Detected gaps: no_specific_example, no_tech_depth

Question 2 (Easy): "Can you explain what a microservice is and why you'd use it?"
Answer Score: 0.75 (good) → Candidate understands fundamentals

Question 3 (Medium): "In your resume, you mentioned a payment service. Walk me through its architecture."
Answer Score: 0.88 (excellent) → Strong technical depth

Question 4 (Hard): "If that payment service needed to handle 10x traffic, what bottlenecks would you anticipate and how would you address them?"
```

---

#### 3.3 Comparative Benchmarking

**Goal**: Tell candidates how they compare to others

**Database Schema**:
```sql
-- New table for anonymized benchmarks
CREATE TABLE evaluation_benchmarks (
    id UUID PRIMARY KEY,
    role_category VARCHAR(50),  -- 'software_engineer', 'ml_engineer', etc.
    experience_level VARCHAR(20),  -- 'junior', 'mid', 'senior', 'staff'
    dimension VARCHAR(50),  -- 'technical_depth', 'clarity', etc.
    score FLOAT,
    anonymized_date DATE,  -- Anonymize to month level
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast percentile calculations
CREATE INDEX idx_benchmarks_role_exp ON evaluation_benchmarks(role_category, experience_level, dimension);
```

**Benchmarking Service**:
```python
# backend/app/services/benchmark_service.py
class BenchmarkService:
    async def calculate_percentile(
        self,
        score: float,
        dimension: str,
        role: str,
        experience: str
    ) -> int:
        """Calculate percentile rank (0-100)"""

        query = """
            SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM evaluation_benchmarks
                WHERE role_category = $1 AND experience_level = $2 AND dimension = $3)
            FROM evaluation_benchmarks
            WHERE role_category = $1
              AND experience_level = $2
              AND dimension = $3
              AND score < $4
        """

        result = await db.fetch_one(query, role, experience, dimension, score)
        return int(result[0])

    async def get_benchmark_insights(
        self,
        session: InterviewSession
    ) -> BenchmarkReport:
        """Generate comparison report"""

        percentiles = {}
        for dimension in ["technical_depth", "clarity", "communication"]:
            avg_score = calculate_avg_dimension_score(session, dimension)
            percentiles[dimension] = await self.calculate_percentile(
                score=avg_score,
                dimension=dimension,
                role=session.role,  # Extracted from resume
                experience=session.experience_level
            )

        return BenchmarkReport(
            percentiles=percentiles,
            insights=self._generate_insights(percentiles),
            sample_size=await self._get_sample_size(session.role, session.experience_level)
        )

    def _generate_insights(self, percentiles: dict) -> list[str]:
        """Natural language insights"""
        insights = []

        for dimension, percentile in percentiles.items():
            if percentile >= 90:
                insights.append(f"Exceptional {dimension} - top 10% of {role}s")
            elif percentile >= 75:
                insights.append(f"Strong {dimension} - above average")
            elif percentile >= 50:
                insights.append(f"Average {dimension} - room for improvement")
            else:
                insights.append(f"Weak {dimension} - focus area for growth")

        return insights
```

**Privacy Considerations**:
- Fully anonymize data (no user IDs stored)
- Aggregate only (minimum 50 samples before showing percentiles)
- Opt-in system (users choose to contribute to benchmark pool)
- Date anonymization (round to month)

**Example Output**:
```json
{
  "benchmark_report": {
    "role": "Senior Software Engineer",
    "experience": "5-7 years",
    "sample_size": 1247,
    "percentiles": {
      "technical_depth": 78,
      "clarity": 45,
      "communication": 62,
      "problem_solving": 91
    },
    "insights": [
      "Strong technical depth - top 25% of senior engineers",
      "Communication clarity below median - consider practicing STAR method",
      "Problem-solving is exceptional - top 10%"
    ],
    "relative_strengths": ["problem_solving", "technical_depth"],
    "relative_weaknesses": ["clarity"]
  }
}
```

---

### Direction 4: Multimodal Interview Experience 🎥

**Status**: Video infrastructure exists, not analyzed
**Effort**: 6-8 weeks
**Impact**: High (unique differentiator)

#### 4.1 Visual Analysis Integration

**Current**: VideoInterviewRoom.tsx captures webcam, but doesn't analyze

**Enhancement**: Real-time visual cues

```python
# backend/app/services/visual_analysis_service.py
import cv2
import mediapipe as mp
from deepface import DeepFace

class VisualAnalysisService:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.pose = mp.solutions.pose.Pose()

    async def analyze_frame(self, frame: bytes) -> VisualCues:
        """Analyze single video frame"""

        img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)

        # 1. Facial expression analysis
        emotion = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = emotion[0]['dominant_emotion']

        # 2. Eye contact tracking
        face_landmarks = self.face_mesh.process(img)
        eye_contact_score = self._calculate_eye_contact(face_landmarks)

        # 3. Posture analysis
        pose_landmarks = self.pose.process(img)
        posture_score = self._calculate_posture(pose_landmarks)

        return VisualCues(
            emotion=dominant_emotion,
            emotion_confidence=emotion[0]['emotion'][dominant_emotion],
            eye_contact_score=eye_contact_score,
            posture_score=posture_score,
            timestamp=time.time()
        )

    def _calculate_eye_contact(self, landmarks) -> float:
        """Estimate eye contact from gaze direction"""
        # Check if eyes are looking at camera (centered)
        # Returns 0.0 (looking away) to 1.0 (direct eye contact)

    def _calculate_posture(self, landmarks) -> float:
        """Assess sitting posture"""
        # Check shoulder alignment, back straightness
        # Returns 0.0 (poor posture) to 1.0 (excellent posture)

    async def generate_visual_feedback(
        self,
        visual_history: list[VisualCues]
    ) -> VisualFeedback:
        """Generate coaching feedback from visual analysis"""

        # Aggregate last 30 seconds of cues
        recent_cues = [c for c in visual_history if time.time() - c.timestamp < 30]

        feedback = []

        # Check for consistent nervousness
        nervous_emotions = ['fear', 'sad', 'angry']
        nervous_ratio = sum(1 for c in recent_cues if c.emotion in nervous_emotions) / len(recent_cues)
        if nervous_ratio > 0.6:
            feedback.append({
                "type": "coaching",
                "message": "Take a deep breath - you're doing fine! Nervousness detected.",
                "severity": "low"
            })

        # Check eye contact
        avg_eye_contact = sum(c.eye_contact_score for c in recent_cues) / len(recent_cues)
        if avg_eye_contact < 0.4:
            feedback.append({
                "type": "coaching",
                "message": "Try to look at the camera when speaking",
                "severity": "medium"
            })

        return VisualFeedback(items=feedback)
```

**Frontend Integration**:
```typescript
// frontend/src/components/interview/VisualCoachingOverlay.tsx
export function VisualCoachingOverlay() {
  const [feedback, setFeedback] = useState<VisualFeedback | null>(null);

  useEffect(() => {
    // Capture frame every 2 seconds
    const interval = setInterval(async () => {
      const frame = captureVideoFrame();
      const response = await fetch('/api/v1/visual/analyze', {
        method: 'POST',
        body: frame
      });
      const result = await response.json();
      setFeedback(result);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="absolute top-4 right-4 space-y-2">
      {feedback?.items.map((item, i) => (
        <CoachingHint key={i} severity={item.severity}>
          {item.message}
        </CoachingHint>
      ))}
    </div>
  );
}
```

**Privacy Toggle**:
```typescript
// User can disable visual analysis
const [visualAnalysisEnabled, setVisualAnalysisEnabled] = useState(false);

// Show clear opt-in message
{!visualAnalysisEnabled && (
  <Alert>
    Enable visual analysis for posture and eye contact coaching?
    <Button onClick={() => setVisualAnalysisEnabled(true)}>Enable</Button>
  </Alert>
)}
```

---

#### 4.2 Advanced Voice Analysis

**Current**: Deepgram transcription only

**Enhancement**: Prosody and speech patterns

```python
# backend/app/services/speech_analysis_service.py
import librosa
import numpy as np

class SpeechAnalysisService:
    async def analyze_audio(self, audio_bytes: bytes) -> SpeechMetrics:
        """Analyze speech patterns beyond transcription"""

        # Load audio
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

        # 1. Speaking rate (words per minute)
        duration = librosa.get_duration(y=y, sr=sr)
        transcript = await deepgram_transcribe(audio_bytes)
        word_count = len(transcript.split())
        wpm = (word_count / duration) * 60

        # 2. Filler word detection
        filler_words = ['um', 'uh', 'like', 'you know', 'so', 'actually']
        filler_count = sum(transcript.lower().count(fw) for fw in filler_words)
        filler_ratio = filler_count / word_count if word_count > 0 else 0

        # 3. Pause analysis
        intervals = librosa.effects.split(y, top_db=20)  # Detect silent parts
        pause_durations = []
        for i in range(len(intervals) - 1):
            pause_start = intervals[i][1] / sr
            pause_end = intervals[i + 1][0] / sr
            pause_duration = pause_end - pause_start
            if pause_duration > 0.3:  # Significant pause
                pause_durations.append(pause_duration)

        avg_pause = np.mean(pause_durations) if pause_durations else 0
        long_pauses = len([p for p in pause_durations if p > 2.0])

        # 4. Pitch variation (engagement indicator)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1])]
        pitch_variance = np.var([p for p in pitch_values if p > 0])

        return SpeechMetrics(
            words_per_minute=wpm,
            filler_word_ratio=filler_ratio,
            average_pause_duration=avg_pause,
            long_pause_count=long_pauses,
            pitch_variance=pitch_variance,
            recommendations=self._generate_speech_recommendations(wpm, filler_ratio, long_pauses)
        )

    def _generate_speech_recommendations(
        self,
        wpm: float,
        filler_ratio: float,
        long_pauses: int
    ) -> list[str]:
        """Generate coaching tips"""
        tips = []

        # Speaking rate
        if wpm > 180:
            tips.append("Try to slow down - speaking at {:.0f} WPM (recommended: 120-150)".format(wpm))
        elif wpm < 100:
            tips.append("You can speak a bit faster - aim for 120-150 WPM")

        # Filler words
        if filler_ratio > 0.05:  # >5% filler words
            tips.append("Reduce filler words ('um', 'uh', 'like') - try pausing instead")

        # Long pauses
        if long_pauses > 3:
            tips.append("Multiple long pauses detected - it's okay to think, but try to keep momentum")

        return tips
```

**Real-time Feedback**:
```json
// WebSocket message type: speech_analysis
{
  "type": "speech_analysis",
  "data": {
    "words_per_minute": 165,
    "filler_word_ratio": 0.08,
    "recommendations": [
      "Reduce filler words ('um', 'uh', 'like') - try pausing instead"
    ]
  },
  "display_mode": "subtle_toast"  // Don't interrupt interview
}
```

---

### Direction 5: Knowledge Graph Enhancement 🕸️

**Status**: Currently using vector search only
**Effort**: 4-5 weeks
**Impact**: Medium-High (better context retrieval)

#### 5.1 Resume Knowledge Graph

**Problem**: Current RAG doesn't capture relationships between skills, projects, companies

**Solution**: Build knowledge graph from resume

```python
# backend/app/services/knowledge_graph_service.py
from neo4j import AsyncGraphDatabase

class ResumeKnowledgeGraph:
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )

    async def build_graph_from_resume(self, resume: ParsedResume, resume_id: str):
        """Construct knowledge graph from parsed resume"""

        async with self.driver.session() as session:
            # Create Person node
            await session.run("""
                CREATE (p:Person {id: $resume_id, name: $name})
            """, resume_id=resume_id, name=resume.name)

            # Create Skill nodes and relationships
            for skill in resume.skills:
                await session.run("""
                    MERGE (s:Skill {name: $skill_name})
                    MATCH (p:Person {id: $resume_id})
                    CREATE (p)-[:HAS_SKILL {proficiency: 'unknown'}]->(s)
                """, skill_name=skill, resume_id=resume_id)

            # Create Experience nodes with relationships
            for exp in resume.experiences:
                await session.run("""
                    CREATE (e:Experience {
                        company: $company,
                        role: $role,
                        start_date: $start_date,
                        end_date: $end_date
                    })
                    MATCH (p:Person {id: $resume_id})
                    CREATE (p)-[:WORKED_AT]->(e)
                """, company=exp.company, role=exp.role,
                     start_date=exp.start_date, end_date=exp.end_date,
                     resume_id=resume_id)

                # Extract skills used in this experience
                for skill in self._extract_skills_from_description(exp.responsibilities):
                    await session.run("""
                        MATCH (e:Experience {company: $company}),
                              (s:Skill {name: $skill})
                        CREATE (e)-[:USED_SKILL]->(s)
                    """, company=exp.company, skill=skill)

            # Create Project nodes
            for project in resume.projects:
                await session.run("""
                    CREATE (pr:Project {
                        name: $name,
                        description: $description
                    })
                    MATCH (p:Person {id: $resume_id})
                    CREATE (p)-[:WORKED_ON]->(pr)
                """, name=project.name, description=project.description,
                     resume_id=resume_id)

                # Link projects to skills
                for skill in self._extract_skills_from_description(project.description):
                    await session.run("""
                        MATCH (pr:Project {name: $project_name}),
                              (s:Skill {name: $skill})
                        CREATE (pr)-[:USES_SKILL]->(s)
                    """, project_name=project.name, skill=skill)

    async def query_skill_context(self, resume_id: str, skill: str) -> SkillContext:
        """Get all context about a specific skill"""

        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (p:Person {id: $resume_id})-[:HAS_SKILL]->(s:Skill {name: $skill})
                OPTIONAL MATCH (e:Experience)-[:USED_SKILL]->(s)
                OPTIONAL MATCH (pr:Project)-[:USES_SKILL]->(s)
                RETURN s, collect(DISTINCT e) as experiences, collect(DISTINCT pr) as projects
            """, resume_id=resume_id, skill=skill)

            record = await result.single()

            return SkillContext(
                skill=skill,
                experiences=[exp['company'] + ' - ' + exp['role'] for exp in record['experiences']],
                projects=[proj['name'] for proj in record['projects']],
                related_skills=await self._get_related_skills(resume_id, skill)
            )

    async def _get_related_skills(self, resume_id: str, skill: str) -> list[str]:
        """Find skills frequently used together"""

        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (s1:Skill {name: $skill})<-[:USED_SKILL]-(e:Experience)-[:USED_SKILL]->(s2:Skill)
                WHERE s1 <> s2
                RETURN s2.name as related_skill, count(*) as frequency
                ORDER BY frequency DESC
                LIMIT 5
            """, skill=skill)

            return [record['related_skill'] async for record in result]

    async def generate_smart_follow_up(
        self,
        resume_id: str,
        mentioned_skill: str,
        current_topic: str
    ) -> str:
        """Generate context-aware follow-up using graph traversal"""

        # Example: If candidate mentions "Redis" in payment service discussion
        skill_context = await self.query_skill_context(resume_id, mentioned_skill)

        # Check if Redis was used in other projects
        if len(skill_context.projects) > 1:
            other_project = skill_context.projects[1]
            return f"You mentioned {mentioned_skill} in {current_topic}. I see you also used it in {other_project}. How did the use cases differ?"

        # Check for related skills
        if skill_context.related_skills:
            related = skill_context.related_skills[0]
            return f"You used {mentioned_skill} alongside {related}. Can you explain how they worked together in your architecture?"

        return None  # Fall back to standard follow-up generation
```

**Benefits**:
1. **Cross-project questions**: "You used Redis in both ProjectA and ProjectB - how did the use cases differ?"
2. **Skill progression**: "I see you used Python in 2020 and still use it now - how has your expertise evolved?"
3. **Consistency checking**: Detect if candidate claims skill but it's not mentioned in any experience
4. **Smart context retrieval**: Get all relevant experiences for a skill in one query

---

#### 5.2 Hybrid Retrieval (Vector + Keyword + Graph)

**Current**: ChromaDB vector search only

**Enhancement**: Multi-strategy retrieval

```python
# backend/app/services/hybrid_retriever.py
class HybridRetriever:
    def __init__(self):
        self.vector_db = ChromaDBEmbedder()  # Existing
        self.elasticsearch = Elasticsearch(["http://localhost:9200"])  # Keyword search
        self.graph_db = ResumeKnowledgeGraph()  # Graph traversal

    async def retrieve(
        self,
        resume_id: str,
        query: str,
        top_k: int = 5
    ) -> list[Chunk]:
        """Combine three retrieval strategies"""

        # 1. Vector search (semantic similarity)
        vector_results = await self.vector_db.search(
            resume_id=resume_id,
            query=query,
            top_k=top_k
        )

        # 2. Keyword search (exact matches)
        keyword_results = await self.elasticsearch.search(
            index=f"resume_{resume_id}",
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "section"],  # Boost content field
                        "type": "best_fields"
                    }
                },
                "size": top_k
            }
        )

        # 3. Graph traversal (relationship-based)
        # Extract entities from query (skills, companies, projects)
        entities = self._extract_entities(query)
        graph_results = []
        for entity in entities:
            context = await self.graph_db.query_skill_context(resume_id, entity)
            graph_results.extend(context.to_chunks())

        # Combine and rerank
        all_results = self._merge_results(vector_results, keyword_results, graph_results)
        reranked = await self._rerank(query, all_results)

        return reranked[:top_k]

    def _merge_results(self, vector, keyword, graph) -> list[Chunk]:
        """Deduplicate and merge results from all sources"""
        seen = set()
        merged = []

        # Assign source scores
        for chunk in vector:
            chunk.scores['vector'] = chunk.similarity_score
        for chunk in keyword:
            chunk.scores['keyword'] = chunk.elasticsearch_score
        for chunk in graph:
            chunk.scores['graph'] = 1.0  # Graph results are highly relevant

        # Merge without duplicates
        for chunk in vector + keyword + graph:
            chunk_id = f"{chunk.resume_id}_{chunk.section}_{chunk.chunk_index}"
            if chunk_id not in seen:
                seen.add(chunk_id)
                merged.append(chunk)

        return merged

    async def _rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Use cross-encoder model to rerank results"""
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Score each chunk against query
        pairs = [[query, chunk.content] for chunk in chunks]
        scores = reranker.predict(pairs)

        # Combine with original scores (weighted average)
        for chunk, rerank_score in zip(chunks, scores):
            chunk.final_score = (
                0.4 * chunk.scores.get('vector', 0) +
                0.3 * chunk.scores.get('keyword', 0) +
                0.2 * chunk.scores.get('graph', 0) +
                0.1 * rerank_score
            )

        return sorted(chunks, key=lambda c: c.final_score, reverse=True)
```

**Example Query Flow**:
```
Query: "Tell me about your experience with microservices and Redis"

1. Vector Search (Semantic):
   - Chunk: "Built distributed payment system using microservices..." (0.92 similarity)
   - Chunk: "Implemented caching layer with Redis..." (0.88 similarity)

2. Keyword Search (Exact):
   - Chunk: "Technologies: Redis, Docker, Kubernetes" (keyword match: "Redis")
   - Chunk: "Migrated monolith to microservices architecture" (keyword match: "microservices")

3. Graph Traversal:
   - Query: MATCH (p)-[:HAS_SKILL]->(s:Skill {name: "Redis"})
   - Found: Used in "Payment Service" project and "Analytics Pipeline" project
   - Retrieved: Full context for both projects

4. Reranking:
   - Cross-encoder scores each chunk's relevance to full query
   - Final ranking combines all scores

Result: Top chunk is "Built distributed payment system using microservices with Redis for session management" (combines both query terms with context)
```

---

### Direction 6: Panel Interview & Multi-Interviewer 👥

**Status**: Currently single interviewer
**Effort**: 3-4 weeks
**Impact**: Medium (unique feature)

#### Concept

Simulate realistic panel interviews with multiple AI interviewers

```python
# backend/app/core/panel_interview.py
from enum import Enum

class InterviewerRole(str, Enum):
    TECHNICAL_BACKEND = "technical_backend"
    TECHNICAL_FRONTEND = "technical_frontend"
    HIRING_MANAGER = "hiring_manager"
    HR = "hr"
    TEAM_LEAD = "team_lead"

class AIInterviewer(BaseModel):
    role: InterviewerRole
    name: str
    focus_areas: list[str]
    personality: str  # "friendly", "challenging", "neutral"

    def get_system_prompt(self) -> str:
        """Role-specific system prompt"""
        prompts = {
            InterviewerRole.TECHNICAL_BACKEND: """
                You are a senior backend engineer. Focus on:
                - System design and architecture
                - Database design and optimization
                - API design and scalability
                - Performance and reliability
                Ask deep technical questions and expect detailed answers.
            """,
            InterviewerRole.HR: """
                You are an HR manager. Focus on:
                - Cultural fit
                - Communication skills
                - Team collaboration
                - Conflict resolution
                Ask behavioral questions using the STAR method.
            """,
            # ... other roles
        }
        return prompts[self.role]

class PanelInterview:
    def __init__(self, session_id: str, interviewers: list[AIInterviewer]):
        self.session_id = session_id
        self.interviewers = interviewers
        self.current_interviewer_idx = 0
        self.round_robin_mode = True  # Interviewers take turns

    async def conduct_interview(self):
        """Run panel interview with multiple interviewers"""

        # Introduction phase
        await self._send_introduction()

        # Each interviewer asks questions
        for interviewer in self.interviewers:
            await self._interviewer_segment(interviewer, num_questions=2)

        # Panel discussion (interviewers "discuss" candidate)
        await self._panel_discussion()

        # Final decision
        await self._consensus_decision()

    async def _interviewer_segment(self, interviewer: AIInterviewer, num_questions: int):
        """One interviewer's question segment"""

        await self.websocket.send_json({
            "type": "interviewer_introduction",
            "interviewer": interviewer.name,
            "role": interviewer.role,
            "message": f"Hi, I'm {interviewer.name}, the {interviewer.role}. I'll be asking you about {', '.join(interviewer.focus_areas)}."
        })

        for _ in range(num_questions):
            # Generate question with role-specific context
            question = await self.llm_service.generate(
                system_prompt=interviewer.get_system_prompt(),
                prompt=self._build_question_prompt(interviewer),
                temperature=0.7
            )

            await self.websocket.send_json({
                "type": "question",
                "interviewer": interviewer.name,
                "content": question
            })

            # Wait for answer
            answer = await self._wait_for_answer()

            # Evaluate (each interviewer has own evaluation criteria)
            evaluation = await self.grilling_engine.evaluate_answer(
                question=question,
                answer=answer,
                resume_context=self.resume_context,
                evaluator_role=interviewer.role  # Pass role for weighted scoring
            )

            # Conditional follow-up
            if evaluation.overall_score < 0.65:
                follow_up = await self._generate_follow_up(interviewer, evaluation)
                await self.websocket.send_json({
                    "type": "follow_up",
                    "interviewer": interviewer.name,
                    "content": follow_up
                })
                await self._wait_for_answer()

    async def _panel_discussion(self):
        """Simulate interviewers discussing candidate"""

        # Collect all evaluations
        all_evaluations = self._get_all_evaluations()

        # Generate inter-interviewer dialogue
        discussion_prompt = f"""
        You are simulating a panel of interviewers discussing a candidate.

        Interviewers:
        {self._format_interviewers()}

        Evaluations:
        {self._format_evaluations(all_evaluations)}

        Generate a realistic discussion between the interviewers about the candidate's performance.
        Each interviewer should speak 2-3 times, expressing their perspective based on their role.
        Format as: "[Interviewer Name]: [Their comment]"
        """

        discussion = await self.llm_service.generate(
            prompt=discussion_prompt,
            temperature=0.8,  # Higher temperature for more natural dialogue
            max_tokens=800
        )

        # Parse and stream discussion
        for line in discussion.split('\n'):
            if ':' in line:
                interviewer_name, comment = line.split(':', 1)
                await self.websocket.send_json({
                    "type": "panel_discussion",
                    "interviewer": interviewer_name.strip(),
                    "comment": comment.strip()
                })
                await asyncio.sleep(2)  # Pause between comments

    async def _consensus_decision(self):
        """Final hiring decision based on all interviewers' input"""

        votes = []
        for interviewer in self.interviewers:
            # Each interviewer's vote weighted by their evaluation
            evaluations = self._get_evaluations_by_interviewer(interviewer)
            avg_score = sum(e.overall_score for e in evaluations) / len(evaluations)

            vote = "hire" if avg_score >= 0.70 else "no_hire"
            votes.append({
                "interviewer": interviewer.name,
                "vote": vote,
                "score": avg_score,
                "reasoning": self._generate_vote_reasoning(interviewer, evaluations)
            })

        # Consensus logic
        hire_votes = sum(1 for v in votes if v['vote'] == 'hire')
        decision = "hire" if hire_votes >= len(votes) / 2 else "no_hire"

        await self.websocket.send_json({
            "type": "panel_decision",
            "decision": decision,
            "votes": votes,
            "summary": self._generate_decision_summary(votes, decision)
        })
```

**Frontend UI**:
```typescript
// frontend/src/components/interview/PanelInterviewRoom.tsx
export function PanelInterviewRoom() {
  const [currentInterviewer, setCurrentInterviewer] = useState<Interviewer | null>(null);
  const [panelDiscussion, setPanelDiscussion] = useState<DiscussionMessage[]>([]);

  return (
    <div className="grid grid-cols-3 gap-4">
      {/* Interviewer Avatars */}
      <div className="col-span-1 space-y-4">
        {interviewers.map(interviewer => (
          <InterviewerCard
            key={interviewer.name}
            interviewer={interviewer}
            isActive={interviewer.name === currentInterviewer?.name}
            evaluationScore={getInterviewerScore(interviewer)}
          />
        ))}
      </div>

      {/* Interview Area */}
      <div className="col-span-2">
        {/* Show who's currently asking */}
        {currentInterviewer && (
          <div className="mb-4 p-4 bg-blue-50 rounded">
            <div className="font-semibold">{currentInterviewer.name}</div>
            <div className="text-sm text-gray-600">{currentInterviewer.role}</div>
          </div>
        )}

        {/* Messages */}
        <ChatHistory messages={messages} />

        {/* Panel Discussion Phase */}
        {isPanelDiscussion && (
          <PanelDiscussionView discussion={panelDiscussion} />
        )}
      </div>
    </div>
  );
}
```

**Configuration Options**:
```typescript
// User selects panel composition
const panelTemplates = {
  "startup": [
    { role: "technical_backend", name: "Alex", personality: "challenging" },
    { role: "ceo", name: "Morgan", personality: "visionary" }
  ],
  "bigtech": [
    { role: "technical_backend", name: "Chris", personality: "neutral" },
    { role: "technical_frontend", name: "Jordan", personality: "friendly" },
    { role: "hiring_manager", name: "Taylor", personality: "neutral" },
    { role: "hr", name: "Sam", personality: "friendly" }
  ],
  "custom": [] // User picks roles manually
};
```

---

### Direction 7: Personalized Prep Plans 📚

**Status**: Not implemented
**Effort**: 2-3 weeks
**Impact**: High (user retention, monetization opportunity)

#### Concept

After interview, generate actionable study plan

```python
# backend/app/services/prep_plan_generator.py
class PrepPlanGenerator:
    def __init__(self):
        self.resource_database = self._load_learning_resources()

    async def generate_plan(self, session: InterviewSession) -> PrepPlan:
        """Generate personalized study plan from interview results"""

        # Analyze weak areas
        weak_areas = self._identify_weak_areas(session.evaluations)
        strong_areas = self._identify_strong_areas(session.evaluations)

        # Generate focus areas with priorities
        focus_areas = []
        for area in weak_areas:
            focus_areas.append(
                FocusArea(
                    skill=area.skill,
                    current_level=area.current_level,
                    target_level=area.target_level,
                    priority=area.priority,  # 1-10
                    estimated_time=area.estimated_hours,
                    resources=self._recommend_resources(area.skill, area.current_level),
                    practice_exercises=self._recommend_exercises(area.skill),
                    milestones=self._generate_milestones(area)
                )
            )

        # Generate practice interview schedule
        practice_schedule = self._generate_practice_schedule(
            weak_areas=weak_areas,
            current_performance=session.average_score
        )

        return PrepPlan(
            focus_areas=focus_areas,
            strong_areas=strong_areas,
            practice_schedule=practice_schedule,
            estimated_total_time=sum(fa.estimated_time for fa in focus_areas),
            target_interview_date=self._calculate_target_date(focus_areas)
        )

    def _identify_weak_areas(self, evaluations: list[AnswerEvaluation]) -> list[WeakArea]:
        """Find skills/dimensions needing improvement"""

        # Aggregate scores by dimension
        dimension_scores = defaultdict(list)
        for eval in evaluations:
            dimension_scores['technical_depth'].append(eval.scores.depth)
            dimension_scores['clarity'].append(eval.scores.clarity)
            dimension_scores['specificity'].append(eval.scores.specificity)
            # ...

        weak_areas = []
        for dimension, scores in dimension_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.65:  # Threshold for "needs improvement"
                weak_areas.append(WeakArea(
                    skill=dimension,
                    current_level=self._score_to_level(avg_score),
                    target_level="advanced",
                    priority=self._calculate_priority(avg_score, dimension),
                    estimated_hours=self._estimate_study_hours(avg_score, dimension)
                ))

        # Also check for specific technical skills mentioned with gaps
        for eval in evaluations:
            if GapType.NO_TECH_DEPTH in eval.gaps:
                # Extract skill from question
                skill = self._extract_skill_from_question(eval.question)
                if skill:
                    weak_areas.append(WeakArea(
                        skill=skill,
                        current_level="beginner",
                        target_level="intermediate",
                        priority=8,
                        estimated_hours=20
                    ))

        return sorted(weak_areas, key=lambda w: w.priority, reverse=True)

    def _recommend_resources(self, skill: str, current_level: str) -> list[LearningResource]:
        """Recommend books, courses, articles"""

        # Knowledge base of resources
        resources_db = {
            "system_design": {
                "beginner": [
                    LearningResource(
                        type="book",
                        title="Designing Data-Intensive Applications",
                        author="Martin Kleppmann",
                        url="https://dataintensive.net/",
                        estimated_hours=40,
                        description="Comprehensive introduction to distributed systems"
                    ),
                    LearningResource(
                        type="course",
                        title="Grokking the System Design Interview",
                        platform="Educative",
                        url="https://educative.io/...",
                        estimated_hours=20
                    )
                ],
                "intermediate": [
                    LearningResource(
                        type="video",
                        title="System Design Primer",
                        platform="YouTube",
                        url="https://github.com/donnemartin/system-design-primer",
                        estimated_hours=10
                    )
                ]
            },
            "behavioral_interview": {
                "beginner": [
                    LearningResource(
                        type="article",
                        title="STAR Method Guide",
                        url="https://...",
                        estimated_hours=2,
                        description="Learn to structure behavioral answers"
                    )
                ]
            }
            # ... extensive resource database
        }

        return resources_db.get(skill, {}).get(current_level, [])

    def _recommend_exercises(self, skill: str) -> list[PracticeExercise]:
        """Recommend hands-on practice"""

        exercises = {
            "system_design": [
                PracticeExercise(
                    title="Design Twitter",
                    description="Design a Twitter-like social media platform handling millions of users",
                    difficulty="medium",
                    estimated_time=2,  # hours
                    key_concepts=["feed generation", "fanout", "caching", "sharding"]
                ),
                PracticeExercise(
                    title="Design URL Shortener",
                    description="Design a URL shortening service like bit.ly",
                    difficulty="easy",
                    estimated_time=1,
                    key_concepts=["hashing", "database design", "redirects"]
                )
            ],
            "clarity": [
                PracticeExercise(
                    title="Record & Review",
                    description="Record yourself answering 5 behavioral questions. Review and identify filler words, unclear statements.",
                    difficulty="easy",
                    estimated_time=1,
                    key_concepts=["self-awareness", "communication"]
                )
            ]
        }

        return exercises.get(skill, [])

    def _generate_practice_schedule(
        self,
        weak_areas: list[WeakArea],
        current_performance: float
    ) -> PracticeSchedule:
        """Recommend frequency of mock interviews"""

        if current_performance < 0.50:
            frequency = "3x per week"
            duration_weeks = 6
            focus = "Build fundamentals with frequent practice"
        elif current_performance < 0.70:
            frequency = "2x per week"
            duration_weeks = 4
            focus = "Refine weak areas with targeted practice"
        else:
            frequency = "1x per week"
            duration_weeks = 2
            focus = "Maintain sharpness with periodic practice"

        return PracticeSchedule(
            frequency=frequency,
            duration_weeks=duration_weeks,
            focus=focus,
            suggested_days=["Tuesday", "Thursday", "Saturday"][:self._parse_frequency(frequency)],
            interview_types=self._suggest_interview_types(weak_areas)
        )
```

**Example Generated Plan**:
```json
{
  "prep_plan": {
    "summary": {
      "estimated_total_time": "60 hours over 6 weeks",
      "target_interview_date": "2026-03-01",
      "current_readiness": "65%",
      "target_readiness": "90%"
    },
    "focus_areas": [
      {
        "skill": "System Design",
        "current_level": "Beginner",
        "target_level": "Intermediate",
        "priority": 10,
        "estimated_time": 30,
        "resources": [
          {
            "type": "book",
            "title": "Designing Data-Intensive Applications",
            "url": "https://dataintensive.net/",
            "estimated_hours": 40,
            "why_recommended": "Your answers lacked depth on distributed systems and scalability"
          },
          {
            "type": "course",
            "title": "Grokking System Design",
            "platform": "Educative",
            "estimated_hours": 20
          }
        ],
        "practice_exercises": [
          {
            "title": "Design URL Shortener",
            "difficulty": "Easy",
            "estimated_time": 1,
            "key_concepts": ["hashing", "database design"]
          },
          {
            "title": "Design Twitter",
            "difficulty": "Medium",
            "estimated_time": 2,
            "key_concepts": ["feed generation", "caching", "sharding"]
          }
        ],
        "milestones": [
          {
            "week": 2,
            "goal": "Complete DDIA chapters 1-5",
            "verification": "Can explain CAP theorem and consistency models"
          },
          {
            "week": 4,
            "goal": "Design 3 systems from scratch",
            "verification": "Can discuss trade-offs for each design"
          },
          {
            "week": 6,
            "goal": "Mock interview focused on system design",
            "verification": "Score >75% on system design questions"
          }
        ]
      },
      {
        "skill": "Communication Clarity",
        "current_level": "Intermediate",
        "target_level": "Advanced",
        "priority": 7,
        "estimated_time": 10,
        "resources": [
          {
            "type": "article",
            "title": "STAR Method Mastery",
            "url": "https://...",
            "estimated_hours": 2
          }
        ],
        "practice_exercises": [
          {
            "title": "Record & Review",
            "description": "Record 5 STAR answers, reduce filler words by 50%",
            "estimated_time": 3
          }
        ]
      }
    ],
    "practice_schedule": {
      "frequency": "2x per week",
      "duration_weeks": 6,
      "suggested_days": ["Tuesday", "Thursday"],
      "interview_types": [
        {
          "week": 1,
          "type": "Technical - Beginner",
          "focus": "Basic system design"
        },
        {
          "week": 2,
          "type": "Behavioral",
          "focus": "STAR method practice"
        },
        {
          "week": 3,
          "type": "Technical - Intermediate",
          "focus": "Distributed systems"
        }
        // ...
      ]
    },
    "strong_areas": [
      "Problem-solving approach",
      "Code quality discussion"
    ],
    "quick_wins": [
      "Practice speaking slower - detected 180 WPM (recommend 120-150)",
      "Reduce filler words - detected 8% ratio (target <3%)",
      "Add specific metrics to all project descriptions"
    ]
  }
}
```

**Frontend Display**:
```typescript
// frontend/src/app/prep-plan/[sessionId]/page.tsx
export default function PrepPlanPage({ params }: { params: { sessionId: string } }) {
  const { prepPlan } = usePrepPlan(params.sessionId);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1>Your Personalized Prep Plan</h1>

      {/* Summary Card */}
      <Card>
        <h2>Overview</h2>
        <p>Estimated Time: {prepPlan.summary.estimated_total_time}</p>
        <p>Current Readiness: {prepPlan.summary.current_readiness}</p>
        <ProgressBar value={65} target={90} />
      </Card>

      {/* Focus Areas */}
      <div className="grid md:grid-cols-2 gap-6 mt-8">
        {prepPlan.focus_areas.map(area => (
          <FocusAreaCard key={area.skill} area={area} />
        ))}
      </div>

      {/* Practice Schedule */}
      <PracticeScheduleCalendar schedule={prepPlan.practice_schedule} />

      {/* Track Progress */}
      <Button onClick={() => markMilestoneComplete(...)}>
        Mark Milestone Complete
      </Button>
    </div>
  );
}
```

**Monetization Opportunity**:
- Free: Basic prep plan (generic resources)
- Premium: Personalized resources, progress tracking, milestone reminders
- Enterprise: Team analytics, custom resource libraries

---

### Direction 8: Industry-Specific Templates 🏢

**Status**: Not implemented
**Effort**: 2-3 weeks (per industry)
**Impact**: Medium (niche market fit)

#### Concept

Tailored interview experiences for different industries

```python
# backend/app/config/industry_templates.py
class IndustryTemplate(BaseModel):
    industry: str
    key_topics: list[str]
    required_gap_types: list[GapType]
    domain_specific_gaps: list[str]
    question_templates: list[str]
    evaluation_weights: dict[str, float]
    compliance_requirements: list[str]

INDUSTRY_TEMPLATES = {
    "fintech": IndustryTemplate(
        industry="fintech",
        key_topics=[
            "transaction processing",
            "payment systems",
            "fraud detection",
            "regulatory compliance (PCI-DSS, SOC2)",
            "double-entry accounting",
            "reconciliation systems"
        ],
        required_gap_types=[
            GapType.NO_SECURITY_CONSIDERATION,  # Must discuss security
            GapType.NO_COMPLIANCE_MENTION,      # NEW: Must mention regulations
            GapType.NO_ERROR_HANDLING           # Critical in financial systems
        ],
        domain_specific_gaps=[
            "no_idempotency_discussion",  # Payment systems must be idempotent
            "no_audit_trail",             # Must discuss audit logging
            "no_data_accuracy",           # Financial accuracy critical
        ],
        question_templates=[
            "Describe a payment system you built. How did you ensure idempotency?",
            "Walk me through how you'd design a fraud detection system.",
            "How have you handled PCI-DSS compliance in your previous role?"
        ],
        evaluation_weights={
            "security_awareness": 1.5,    # 50% more weight on security
            "accuracy_emphasis": 1.4,     # High weight on data accuracy
            "technical_depth": 1.2
        },
        compliance_requirements=["PCI-DSS", "SOC2", "GDPR"]
    ),

    "healthcare": IndustryTemplate(
        industry="healthcare",
        key_topics=[
            "HIPAA compliance",
            "PHI (Protected Health Information)",
            "HL7/FHIR standards",
            "medical data security",
            "audit logging",
            "patient privacy"
        ],
        required_gap_types=[
            GapType.NO_PRIVACY_CONSIDERATION,
            GapType.NO_COMPLIANCE_MENTION,
            GapType.NO_SECURITY_CONSIDERATION
        ],
        domain_specific_gaps=[
            "no_phi_handling",           # Must discuss PHI handling
            "no_consent_management",     # Patient consent critical
            "no_data_retention_policy"   # Healthcare has strict retention rules
        ],
        question_templates=[
            "How have you ensured HIPAA compliance in your applications?",
            "Describe how you'd design a system to handle PHI securely.",
            "What's your experience with HL7 or FHIR standards?"
        ],
        evaluation_weights={
            "privacy_awareness": 1.6,
            "compliance_knowledge": 1.5,
            "security_awareness": 1.4
        },
        compliance_requirements=["HIPAA", "HITECH", "GDPR"]
    ),

    "ecommerce": IndustryTemplate(
        industry="ecommerce",
        key_topics=[
            "inventory management",
            "recommendation systems",
            "search optimization",
            "checkout flow",
            "performance at scale",
            "A/B testing"
        ],
        domain_specific_gaps=[
            "no_conversion_metrics",     # Must discuss conversion rates
            "no_performance_impact",     # Page speed affects revenue
            "no_ab_testing_mention"      # E-commerce relies on experimentation
        ],
        question_templates=[
            "How have you optimized checkout conversion rates?",
            "Describe a recommendation system you built or worked with.",
            "How do you approach A/B testing for new features?"
        ],
        evaluation_weights={
            "metrics_orientation": 1.4,  # E-commerce is metrics-driven
            "performance_awareness": 1.3,
            "user_experience": 1.2
        }
    ),

    "startup": IndustryTemplate(
        industry="startup",
        key_topics=[
            "rapid iteration",
            "MVP development",
            "ambiguity tolerance",
            "ownership mentality",
            "technical debt trade-offs",
            "wearing multiple hats"
        ],
        domain_specific_gaps=[
            "no_ownership_example",      # Startups value ownership
            "no_ambiguity_handling",     # Must handle unclear requirements
            "no_tradeoff_discussion"     # MVP means making trade-offs
        ],
        question_templates=[
            "Tell me about a time you had to make a decision with incomplete information.",
            "How do you balance speed vs. code quality when building an MVP?",
            "Describe a project you owned end-to-end."
        ],
        evaluation_weights={
            "autonomy": 1.5,
            "adaptability": 1.4,
            "pragmatism": 1.3
        }
    )
}
```

**Usage**:
```python
# backend/app/api/routes/session.py
@router.post("/sessions")
async def create_session(
    resume_id: str,
    mode: InterviewMode,
    industry: str = "general"  # NEW parameter
):
    # Load industry template
    template = INDUSTRY_TEMPLATES.get(industry)

    # Configure grilling engine with industry-specific weights
    grilling_engine = GrillingEngine(
        model_type="api",
        evaluation_weights=template.evaluation_weights if template else None,
        additional_gap_types=template.domain_specific_gaps if template else []
    )

    # Generate questions from industry-specific templates
    questions = await interview_agent.generate_questions(
        resume_id=resume_id,
        mode=mode,
        industry_template=template
    )

    return {"session_id": session.session_id, "industry": industry}
```

**Frontend Selection**:
```typescript
// frontend/src/app/upload/page.tsx
<Select value={industry} onValueChange={setIndustry}>
  <option value="general">General Tech</option>
  <option value="fintech">Fintech / Banking</option>
  <option value="healthcare">Healthcare / MedTech</option>
  <option value="ecommerce">E-Commerce / Retail</option>
  <option value="startup">Startup / Early Stage</option>
  <option value="enterprise">Enterprise Software</option>
</Select>
```

---

### Direction 9: External Platform Integrations 🔗

**Status**: Not implemented
**Effort**: 3-4 weeks
**Impact**: Medium (user acquisition, convenience)

#### 9.1 LinkedIn Integration

```python
# backend/app/services/linkedin_service.py
from linkedin_api import Linkedin

class LinkedInService:
    async def import_profile(self, access_token: str) -> ParsedResume:
        """Import LinkedIn profile and convert to resume"""

        api = Linkedin('', '', authenticate=True)
        profile = api.get_profile(access_token)

        # Convert LinkedIn data to ParsedResume format
        resume = ParsedResume(
            name=profile['firstName'] + ' ' + profile['lastName'],
            email=profile.get('emailAddress', ''),
            phone='',  # LinkedIn doesn't provide
            summary=profile.get('summary', ''),
            skills=[skill['name'] for skill in profile.get('skills', [])],
            experiences=[
                Experience(
                    company=exp['companyName'],
                    role=exp['title'],
                    start_date=self._parse_linkedin_date(exp['timePeriod']['startDate']),
                    end_date=self._parse_linkedin_date(exp['timePeriod'].get('endDate')),
                    responsibilities=exp.get('description', '').split('\n')
                )
                for exp in profile.get('positions', [])
            ],
            education=[
                Education(
                    institution=edu['schoolName'],
                    degree=edu.get('degreeName', ''),
                    field_of_study=edu.get('fieldOfStudy', ''),
                    start_date=self._parse_linkedin_date(edu['timePeriod']['startDate']),
                    end_date=self._parse_linkedin_date(edu['timePeriod'].get('endDate'))
                )
                for edu in profile.get('education', [])
            ]
        )

        return resume

    async def verify_consistency(
        self,
        uploaded_resume: ParsedResume,
        linkedin_profile: ParsedResume
    ) -> ConsistencyReport:
        """Check for discrepancies between resume and LinkedIn"""

        discrepancies = []

        # Check job titles
        for resume_exp in uploaded_resume.experiences:
            matching_linkedin = next(
                (exp for exp in linkedin_profile.experiences
                 if exp.company == resume_exp.company),
                None
            )
            if matching_linkedin and matching_linkedin.role != resume_exp.role:
                discrepancies.append({
                    "type": "job_title_mismatch",
                    "company": resume_exp.company,
                    "resume_title": resume_exp.role,
                    "linkedin_title": matching_linkedin.role
                })

        # Check dates
        # Check skills
        # ...

        return ConsistencyReport(
            is_consistent=len(discrepancies) == 0,
            discrepancies=discrepancies,
            confidence_score=self._calculate_consistency_score(discrepancies)
        )
```

---

#### 9.2 LeetCode Integration

```python
# backend/app/services/coding_challenge_service.py
class CodingChallengeService:
    async def insert_coding_challenge(
        self,
        session_id: str,
        difficulty: str = "medium"
    ):
        """Insert a coding challenge mid-interview"""

        # Select challenge based on resume skills
        challenge = await self._select_appropriate_challenge(session_id, difficulty)

        await websocket.send_json({
            "type": "coding_challenge",
            "challenge": {
                "title": challenge.title,
                "description": challenge.description,
                "examples": challenge.examples,
                "constraints": challenge.constraints,
                "starter_code": challenge.starter_code,
                "language_options": ["python", "javascript", "java", "cpp"]
            },
            "time_limit_minutes": 30
        })

    async def evaluate_code_submission(
        self,
        session_id: str,
        code: str,
        language: str
    ) -> CodeEvaluation:
        """Run test cases against submitted code"""

        # Use Judge0 API or custom sandbox
        test_results = await self._run_test_cases(code, language, challenge.test_cases)

        # Analyze code quality
        quality_metrics = await self._analyze_code_quality(code, language)

        return CodeEvaluation(
            test_cases_passed=test_results.passed,
            test_cases_total=test_results.total,
            time_complexity=quality_metrics.time_complexity,
            space_complexity=quality_metrics.space_complexity,
            code_quality_score=quality_metrics.overall_score,
            feedback=self._generate_code_feedback(test_results, quality_metrics)
        )
```

**Frontend - Monaco Editor Integration**:
```typescript
// frontend/src/components/interview/CodeEditor.tsx
import Editor from '@monaco-editor/react';

export function CodeEditor({ challenge }: { challenge: CodingChallenge }) {
  const [code, setCode] = useState(challenge.starter_code);
  const [language, setLanguage] = useState('python');

  const handleSubmit = async () => {
    const result = await fetch('/api/v1/coding/submit', {
      method: 'POST',
      body: JSON.stringify({ code, language, session_id })
    });

    const evaluation = await result.json();
    showResults(evaluation);
  };

  return (
    <div className="h-screen flex flex-col">
      <div className="flex justify-between p-4">
        <h2>{challenge.title}</h2>
        <Select value={language} onValueChange={setLanguage}>
          <option value="python">Python</option>
          <option value="javascript">JavaScript</option>
          <option value="java">Java</option>
        </Select>
      </div>

      <div className="flex-1">
        <Editor
          height="100%"
          language={language}
          value={code}
          onChange={(value) => setCode(value || '')}
          theme="vs-dark"
        />
      </div>

      <div className="p-4">
        <Button onClick={handleSubmit}>Submit Code</Button>
      </div>
    </div>
  );
}
```

---

#### 9.3 Calendar Integration

```python
# backend/app/services/calendar_service.py
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

class CalendarService:
    async def schedule_practice_interview(
        self,
        user_id: str,
        datetime: str,
        interview_type: str
    ):
        """Schedule practice interview on user's calendar"""

        credentials = await self._get_user_credentials(user_id)
        service = build('calendar', 'v3', credentials=credentials)

        event = {
            'summary': f'Practice Interview - {interview_type}',
            'description': f'Scheduled practice interview via Resume Griller\nType: {interview_type}\nPrepare by reviewing your weak areas.',
            'start': {
                'dateTime': datetime,
                'timeZone': 'America/Los_Angeles',
            },
            'end': {
                'dateTime': self._add_hours(datetime, 1),
                'timeZone': 'America/Los_Angeles',
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},  # 1 day before
                    {'method': 'popup', 'minutes': 30},       # 30 min before
                ],
            },
        }

        event = service.events().insert(calendarId='primary', body=event).execute()
        return event['htmlLink']
```

---

### Direction 10: Benchmarking & Leaderboards 🏆

**Status**: Not implemented
**Effort**: 2 weeks
**Impact**: Medium (gamification, engagement)

#### Anonymized Leaderboards

```python
# backend/app/services/leaderboard_service.py
class LeaderboardService:
    async def get_leaderboard(
        self,
        role: str,
        experience: str,
        timeframe: str = "all_time"  # "week", "month", "all_time"
    ) -> Leaderboard:
        """Get anonymized leaderboard"""

        query = """
            SELECT
                anonymized_id,
                AVG(overall_score) as avg_score,
                COUNT(DISTINCT session_id) as interview_count,
                AVG(technical_depth) as avg_technical,
                AVG(clarity) as avg_clarity
            FROM anonymized_sessions
            WHERE role_category = $1
              AND experience_level = $2
              AND created_at > $3
            GROUP BY anonymized_id
            HAVING COUNT(DISTINCT session_id) >= 3  -- Min 3 interviews to appear
            ORDER BY avg_score DESC
            LIMIT 100
        """

        timeframe_filter = {
            "week": "NOW() - INTERVAL '7 days'",
            "month": "NOW() - INTERVAL '30 days'",
            "all_time": "'1970-01-01'"
        }

        results = await db.fetch_all(query, role, experience, timeframe_filter[timeframe])

        return Leaderboard(
            role=role,
            experience=experience,
            timeframe=timeframe,
            entries=[
                LeaderboardEntry(
                    rank=idx + 1,
                    anonymized_id=row['anonymized_id'],
                    avg_score=row['avg_score'],
                    interview_count=row['interview_count'],
                    badges=self._calculate_badges(row)
                )
                for idx, row in enumerate(results)
            ],
            user_rank=await self._get_user_rank(user_id, role, experience) if user_id else None
        )

    def _calculate_badges(self, row: dict) -> list[str]:
        """Award badges for achievements"""
        badges = []

        if row['avg_score'] >= 0.90:
            badges.append("🏆 Top Performer")
        if row['avg_technical'] >= 0.85:
            badges.append("💻 Tech Expert")
        if row['avg_clarity'] >= 0.90:
            badges.append("🗣️ Clear Communicator")
        if row['interview_count'] >= 10:
            badges.append("🔥 Dedicated Practicer")

        return badges
```

**Privacy-First Approach**:
- Anonymous IDs (e.g., "Candidate_7f3a2")
- Opt-in only (users choose to appear)
- No identifying information displayed
- Minimum threshold (3 interviews) to prevent de-anonymization

---

## 📅 Prioritized Roadmap

### Phase 1: Foundation & Stability (Q1 2026) - 6 weeks

**Goal**: Make system production-ready

| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Implement PostgreSQL session storage | 🔴 Critical | 1 week | Backend |
| Add comprehensive test suite (70%+ coverage) | 🔴 Critical | 2 weeks | Backend |
| Clean up duplicate code | 🟡 High | 2 days | Backend |
| Add authentication & rate limiting | 🟡 High | 1 week | Backend |
| Replace print() with structured logging | 🟡 High | 3 days | Backend |
| Add Docker Compose for deployment | 🟡 High | 3 days | DevOps |
| Automatically create data directories | 🟢 Medium | 1 day | Backend |

**Deliverables**:
- ✅ Zero data loss (persistent storage)
- ✅ Secure (auth + rate limits)
- ✅ Testable (70%+ coverage)
- ✅ Deployable (Docker Compose)
- ✅ Observable (structured logs)

---

### Phase 2: LangGraph Architecture (Q2 2026) - 4 weeks

**Goal**: Multi-agent refactor with checkpoints

| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Design LangGraph state machine | 🔴 Critical | 3 days | Architect |
| Implement checkpoint infrastructure | 🔴 Critical | 1 week | Backend |
| Migrate InterviewAgent to LangGraph | 🔴 Critical | 1 week | Backend |
| Integrate with WebSocket handler | 🔴 Critical | 4 days | Backend |
| Create QuestionGenerator agent | 🟡 High | 3 days | Backend |
| Create Evaluator agent | 🟡 High | 3 days | Backend |
| Create FollowUp agent | 🟡 High | 3 days | Backend |
| Create Summary agent (NEW) | 🟢 Medium | 2 days | Backend |
| Integration testing | 🟡 High | 3 days | QA |

**Deliverables**:
- ✅ State persists across restarts
- ✅ Interview resumable at any point
- ✅ Visual debugging with LangGraph Studio
- ✅ Modular agent architecture

---

### Phase 3: Intelligence Upgrades (Q2-Q3 2026) - 6 weeks

**Goal**: Best-in-class grilling intelligence

| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Domain-specific scoring models | 🟡 High | 1 week | ML/Backend |
| Adaptive difficulty system | 🟡 High | 1 week | Backend |
| Comparative benchmarking | 🟢 Medium | 1 week | Backend |
| Resume knowledge graph (Neo4j) | 🟢 Medium | 1 week | Backend |
| Hybrid retrieval (vector + keyword + graph) | 🟢 Medium | 1 week | Backend |
| Industry-specific templates (Fintech, Healthcare) | 🟢 Medium | 1 week | Content + Backend |

**Deliverables**:
- ✅ Role-aware evaluation (frontend vs backend vs ML)
- ✅ Questions adapt to candidate performance
- ✅ Percentile rankings vs peers
- ✅ Smarter context retrieval

---

### Phase 4: Multimodal Features (Q3 2026) - 8 weeks

**Goal**: Unique differentiators

| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Visual analysis integration (facial, posture) | 🟢 Medium | 2 weeks | ML/Backend |
| Advanced speech analysis (prosody, filler words) | 🟢 Medium | 1 week | ML/Backend |
| Real-time coaching overlay | 🟢 Medium | 1 week | Frontend |
| Panel interview mode | 🟢 Medium | 2 weeks | Backend + Frontend |
| Personalized prep plan generator | 🟡 High | 2 weeks | Backend |
| Progress tracking dashboard | 🟢 Medium | 1 week | Frontend |

**Deliverables**:
- ✅ Eye contact & posture coaching
- ✅ Speaking rate & filler word detection
- ✅ Multi-interviewer simulation
- ✅ Actionable study plans

---

### Phase 5: Ecosystem & Growth (Q4 2026) - 6 weeks

**Goal**: User acquisition & retention

| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| LinkedIn integration | 🟢 Medium | 1 week | Backend |
| LeetCode/coding challenge integration | 🟢 Medium | 2 weeks | Backend + Frontend |
| Calendar integration (Google/Outlook) | 🟢 Low | 3 days | Backend |
| Anonymized leaderboards | 🟢 Medium | 1 week | Backend + Frontend |
| Industry templates (3+ industries) | 🟢 Medium | 2 weeks | Content + Backend |

**Deliverables**:
- ✅ One-click resume import from LinkedIn
- ✅ Live coding challenges mid-interview
- ✅ Automated practice scheduling
- ✅ Gamification via leaderboards

---

## ⚡ Quick Wins (Can Do Immediately)

These can be completed in 1-2 days each with high impact:

### 1. Clean Up Duplicate Code (2 hours)
```bash
# Delete orphaned code
rm -rf scripts/backend/
rm backend/app/core/resume_parser.py  # Use rag/resume_parser.py instead

# Update imports
find backend/ -type f -name "*.py" -exec sed -i 's/from backend.app.core.resume_parser/from rag.resume_parser/g' {} +
```

### 2. Auto-Create Data Directories (30 minutes)
```python
# backend/app/main.py
from pathlib import Path

@app.on_event("startup")
async def startup_event():
    # Create required directories
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    Path("data/chromadb").mkdir(parents=True, exist_ok=True)
    logger.info("Data directories initialized")
```

### 3. Add Basic Logging (3 hours)
```python
# backend/app/core/logging_config.py
import structlog

def configure_logging():
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ]
    )

# Replace all print() statements
# Old: print(f"[GrillingEngine] Evaluating...")
# New: logger.info("evaluating_answer", component="grilling_engine")
```

### 4. Add Docker Compose (4 hours)
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/resume_griller
    depends_on:
      - postgres
      - redis

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: resume_griller
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### 5. Add Health Check Improvements (1 hour)
```python
# backend/app/api/routes/health.py
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "llm_mode": settings.LLM_MODE,
        "llm_provider": settings.LLM_PROVIDER,
        "voice_enabled": settings.VOICE_ENABLED,
        "custom_model_available": await check_custom_model(),  # Ping vLLM
        "database_connected": await check_database(),
        "redis_connected": await check_redis()
    }
```

### 6. Add Rate Limiting (2 hours)
```python
# backend/app/middleware/rate_limit.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Apply to expensive endpoints
@router.post("/sessions")
@limiter.limit("5/minute")  # 5 session creations per minute
async def create_session(...):
    pass

@router.post("/sessions/{session_id}/answer")
@limiter.limit("10/minute")  # 10 answers per minute
async def submit_answer(...):
    pass
```

---

## 🔧 Technical Debt Summary

### Critical Debt (Blocks Production)
1. **In-memory session storage** - Data loss on restart
2. **No tests** - Cannot safely refactor
3. **No authentication** - Security risk

### High-Priority Debt (Should Fix Soon)
1. **Duplicate code** - `backend/app/core/resume_parser.py` + `scripts/backend/`
2. **Print statements** - Should use structured logging
3. **Hardcoded values** - Move to config (e.g., `SUFFICIENT_SCORE_THRESHOLD = 0.58`)
4. **No Docker** - Difficult to deploy

### Medium-Priority Debt (Refactoring Opportunities)
1. **GrillingEngine complexity** - 973 lines, could split into modules
2. **LLM Service sprawl** - 828 lines, could extract providers
3. **WebSocket handler length** - 776 lines, could extract handlers

---

## 🏗️ Architecture Evolution

### Current Architecture (As-Is)
```
FastAPI Backend + React Frontend + ChromaDB
├─ In-memory session store (PROBLEM: data loss)
├─ 6 LLM providers (working)
├─ Grilling engine with 18 gap types (excellent)
├─ WebSocket for real-time (working)
└─ Voice services (working)
```

### Target Architecture (To-Be after Phase 2)
```
LangGraph Multi-Agent System
├─ PostgreSQL session store + checkpoints (persistent)
├─ Agent: QuestionGenerator
├─ Agent: Evaluator (reuses GrillingEngine)
├─ Agent: FollowUpGenerator
├─ Agent: SummaryGenerator (NEW)
├─ Redis for caching
└─ Neo4j for knowledge graph (Phase 3)
```

### Long-Term Vision (6+ months)
```
Microservices Architecture
├─ API Gateway (authentication, rate limiting)
├─ Interview Service (LangGraph agents)
├─ Voice Service (STT/TTS)
├─ RAG Service (vector + graph retrieval)
├─ Analytics Service (benchmarking, leaderboards)
└─ Shared: PostgreSQL, Redis, Neo4j, ChromaDB
```

---

## 📈 Success Metrics

### Current Baseline (Estimate)
- **Session completion rate**: Unknown (no persistence = can't measure)
- **Average interview score**: Unknown (no analytics)
- **User retention**: Unknown (no auth/tracking)

### Target Metrics (After Phase 1)
- **Session completion rate**: >80%
- **Data loss rate**: 0% (persistent storage)
- **System uptime**: >99.5%
- **Test coverage**: >70%

### Target Metrics (After Phase 4)
- **User retention (7-day)**: >40%
- **Interview quality score** (user rating): >4.2/5
- **Prep plan adoption**: >60% of users view plan
- **Practice interview frequency**: 2x per week average

---

## 💡 Innovative Ideas (Future Exploration)

1. **AI Interview Coach Companion**
   - Real-time GPT-4 analysis during interview
   - Suggests better phrasing on the fly
   - Post-interview: "You could have said X instead of Y"

2. **Interview Replay with Analysis**
   - Record full interview (video + audio)
   - Timestamp key moments (gaps detected, excellent answers)
   - Generate highlight reel of best/worst moments

3. **Collaborative Interview Prep**
   - Pair users for mock interviews
   - One user interviews, other observes + scores
   - Switch roles

4. **Company-Specific Interview Prep**
   - Templates for FAANG (Google, Meta, Amazon)
   - Based on Glassdoor interview questions
   - "Simulate a Google L4 backend interview"

5. **Voice Cloning for Realistic Interviewers**
   - Clone voice of famous interviewers
   - "Practice with a Linus Torvalds-style interviewer"

6. **AR/VR Interview Simulation**
   - Full immersive interview room
   - Body language practice in VR
   - Metaverse interview experiences

---

## 📝 Notes for Other AI Agents

### Key Context
- **Latest commit** adds `langgraph` dependencies → Major refactor incoming
- **Biggest issue**: In-memory session storage → Fix this FIRST
- **Strongest component**: GrillingEngine (973 lines, 18 gap types) → Don't break this
- **Most complex**: LLM Service (828 lines, 6 providers) → Handle with care
- **Duplicate code**: `scripts/backend/` + `backend/app/core/resume_parser.py` → Safe to delete

### Before Making Changes
1. Read `CLAUDE.md` (project overview)
2. Check `backend/app/core/grilling_engine.py` (core logic)
3. Check `.env.example` (required config)
4. Run tests: `PYTHONPATH=. uv run pytest tests/` (currently minimal)

### Common Pitfalls
- Don't use `backend/app/core/resume_parser.py` → Use `rag/resume_parser.py`
- Don't forget `PYTHONPATH=.` when running Python scripts
- WebSocket handler is long (776 lines) → Read carefully before modifying
- Sessions are in-memory → Any restart loses data (until Phase 1 complete)

### Safe Areas to Modify
- Frontend components (React 19 + Next.js 16)
- Voice services (STT/TTS wrappers)
- RAG pipeline (well-isolated)
- Adding new LLM providers

### Risky Areas (Test Thoroughly)
- GrillingEngine evaluation logic
- WebSocket message handling
- Session state management
- LLM service provider switching

---

## 🎓 Learning Resources

For developers new to this codebase:

1. **LangGraph** (for Phase 2): https://langchain-ai.github.io/langgraph/
2. **ChromaDB**: https://docs.trychroma.com/
3. **FastAPI WebSockets**: https://fastapi.tiangolo.com/advanced/websockets/
4. **React 19 Docs**: https://react.dev/
5. **Pydantic**: https://docs.pydantic.dev/

---

**End of Roadmap**

This document is a living roadmap. Update it as priorities shift and features are completed.

**Last Updated**: 2026-01-16
**Next Review**: After Phase 1 completion (target: 2026-03-01)
