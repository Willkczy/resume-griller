// frontend/src/types/index.ts

// ============== Resume Types ==============

export interface ResumeUploadResponse {
  resume_id: string;
  filename: string;
  chunks_created: number;
  sections: string[];
  message: string;
}

export interface ResumeSummary {
  resume_id: string;
  name: string | null;
  total_chunks: number;
  sections: string[];
  skills: string[];
  experience_count: number;
  education_count: number;
}

// ============== Session Types ==============

export type InterviewMode = 'hr' | 'tech' | 'mixed';
export type ModelType = 'api' | 'custom';
export type SessionStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled';

export interface SessionCreateRequest {
  resume_id: string;
  mode: InterviewMode;
  model_type: ModelType;
  num_questions?: number;
  max_follow_ups?: number;
  focus_areas?: string[];
}

export interface SessionResponse {
  session_id: string;
  resume_id: string;
  mode: string;
  model_type: ModelType;
  status: SessionStatus;
  current_question_index: number;
  total_questions: number;
  questions_asked: number;
  created_at: string;
  updated_at: string;
}

export interface ConversationMessage {
  role: 'interviewer' | 'candidate' | 'system';
  content: string;
  timestamp: string;
  is_follow_up: boolean;
  question_id?: string | null;
  parent_question_id?: string | null;
  answer_id?: string | null;
}

export type SessionDetail = InterviewSnapshot;

export interface SessionSummary {
  session_id: string;
  resume_id: string;
  mode: string;
  status: string;
  questions_asked: number;
  total_questions: number;
  answers_given: number;
  follow_ups_asked: number;
  duration_seconds: number;
  conversation_length: number;
}

// ============== Interview Response Types ==============

export interface ModelExecution {
  schema_version: 'model-execution-v1';
  call_id: string;
  operation: 'question_generation' | 'evaluation' | 'follow_up_generation'
    | 'resume_parsing' | 'resume_summary' | 'question_context';
  provider: 'anthropic' | 'openai' | 'gemini' | 'groq' | 'custom' | null;
  configured_model: string | null;
  model: string | null;
  provider_request_id: string | null;
  trace_id: string | null;
  usage: {
    input_tokens: number | null;
    output_tokens: number | null;
    total_tokens: number | null;
    cache_read_input_tokens: number | null;
    cache_creation_input_tokens: number | null;
  };
  cost_usd: number | null;
  finish_reason: string | null;
  latency_ms: number;
  prompt_version: string;
  rubric_version: string | null;
  output_schema_version: string | null;
  outcome: 'valid' | 'insufficient_evidence' | 'invalid_output' | 'provider_failure';
  error_code: string | null;
}

export type InterviewAction = 'start' | 'answer' | 'skip' | 'end';
export type AnswerStatus = 'none' | 'unscored' | 'accepted';

interface EvaluationIdentity {
  question_id: string | null;
  parent_question_id: string | null;
  answer_id: string | null;
}

export type GapType =
  | 'no_specific_example' | 'no_alternatives' | 'unclear_statement'
  | 'contradictory' | 'missing_tacit' | 'too_generic' | 'too_long_unfocused'
  | 'unexplained_jargon' | 'no_tech_depth' | 'wrong_scope' | 'solution_not_exp'
  | 'mixed_topics' | 'vague_ambiguous' | 'no_concrete_meaning' | 'no_metrics'
  | 'unclear_role' | 'no_outcome' | 'resume_inconsistent';

export interface ValidEvaluation extends EvaluationIdentity {
  outcome: 'valid';
  follow_up_policy_version: string | null;
  follow_up_failure: GenerationOutcome | null;
  is_sufficient: boolean;
  score: number;
  detailed_scores: {
    relevancy: number;
    clarity: number;
    informativeness: number;
    specificity: number;
    quantification: number;
    depth: number;
    completeness: number;
    overall: number;
  };
  gap_analysis: {
    detected_gaps: GapType[];
    gap_details: Partial<Record<GapType, string>>;
    severity: number;
    priority_gap: GapType | null;
  };
  missing_elements: string[];
  strengths: string[];
  suggested_follow_up: string | null;
  reasoning: string;
}

export interface InsufficientEvidence extends EvaluationIdentity {
  outcome: 'insufficient_evidence';
  reasoning: string;
  missing_elements: string[];
  suggested_follow_up: string | null;
}

export interface InvalidOutput extends EvaluationIdentity {
  outcome: 'invalid_output';
  code: 'invalid_json' | 'schema_violation';
}

export interface ProviderFailure extends EvaluationIdentity {
  outcome: 'provider_failure';
  code: 'timeout' | 'rate_limited' | 'authentication' | 'unavailable' | 'unknown';
}

export interface UnsupportedMode extends EvaluationIdentity {
  outcome: 'unsupported_mode';
  code: 'evaluation_contract_unavailable';
}

export type AnswerEvaluation = ValidEvaluation | InsufficientEvidence | InvalidOutput
  | ProviderFailure | UnsupportedMode;

export type GenerationOutcome = Pick<InvalidOutput, 'outcome' | 'code'>
  | Pick<ProviderFailure, 'outcome' | 'code'> | Pick<UnsupportedMode, 'outcome' | 'code'>;

export interface GenerationFailure {
  operation: 'question_generation' | 'follow_up_generation';
  outcome: GenerationOutcome;
}

/** Authoritative checkpoint projection shared by REST and WebSocket commands. */
export interface InterviewSnapshot {
  model_executions: ModelExecution[];
  session_id: string;
  resume_id: string;
  mode: string;
  model_type: string;
  status: SessionStatus;
  current_question: string | null;
  current_question_id: string | null;
  parent_question_id: string | null;
  is_follow_up: boolean;
  total_questions: number;
  follow_up_count: number;
  max_follow_ups: number;
  created_at: string;
  updated_at: string;
  conversation: ConversationMessage[];
  evaluation: AnswerEvaluation | null;
  generation_failure: GenerationFailure | null;
  draft_answer: string | null;
  answer_status: AnswerStatus;
  allowed_actions: InterviewAction[];
  current_question_index: number;
  question_number?: number | null;
}

export interface InterviewResponse extends InterviewSnapshot {
  type: 'question' | 'follow_up' | 'complete' | 'error';
  content: string;
  metadata?: Record<string, unknown> | null;
}

// ============== WebSocket Types ==============

export type WSMessageType = 
  | 'start' 
  | 'answer' 
  | 'answer_audio' 
  | 'skip' 
  | 'end' 
  | 'ping'
  | 'question'
  | 'follow_up'
  | 'complete'
  | 'error'
  | 'connected'
  | 'transcript'
  | 'pong';

export interface WSMessageData extends Partial<InterviewSnapshot> {
  session_id?: string;
  resume_id?: string;
  mode?: InterviewMode;
  model_type?: ModelType;
  status?: SessionStatus;
  current_question?: string | null;
  current_question_id?: string | null;
  question_id?: string | null;
  parent_question_id?: string | null;
  is_follow_up?: boolean;
  follow_up_count?: number;
  question_number?: number | null;
  total_questions?: number;
  confidence?: number;
  priority_gap?: GapType | null;
  follow_up_policy_version?: string | null;
  summary?: Record<string, unknown> | null;
  error_code?: string | null;
  status_code?: number | null;
  error?: string | null;
  [key: string]: unknown;
}

export interface WSClientMessageData {
  num_questions?: number;
  voice_enabled?: boolean;
  [key: string]: unknown;
}

export interface WSMessage {
  type: WSMessageType;
  content?: string;
  timestamp?: string;
  data?: WSMessageData;
  error?: string;
  audio_base64?: string;
}

export interface WSClientMessage {
  type: 'start' | 'answer' | 'answer_audio' | 'skip' | 'end' | 'ping';
  content?: string;
  data?: WSClientMessageData;
}

// ============== Voice Types ==============

export interface VoiceStatus {
  enabled: boolean;
  stt_provider: string;
  tts_provider: string;
  stt_available: boolean;
  tts_available: boolean;
}

export interface TranscribeResponse {
  text: string;
  confidence: number;
  is_final: boolean;
  duration_seconds: number;
}
