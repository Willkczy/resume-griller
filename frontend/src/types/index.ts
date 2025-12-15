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
export type SessionStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled';

export interface SessionCreateRequest {
  resume_id: string;
  mode: InterviewMode;
  num_questions?: number;
  max_follow_ups?: number;
  focus_areas?: string[];
}

export interface SessionResponse {
  session_id: string;
  resume_id: string;
  mode: string;
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
}

export interface SessionDetail extends SessionResponse {
  current_question: string | null;
  follow_up_count: number;
  max_follow_ups: number;
  conversation: ConversationMessage[];
}

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

export interface InterviewResponse {
  type: 'question' | 'follow_up' | 'complete' | 'error';
  content: string;
  question_number?: number;
  total_questions?: number;
  evaluation?: AnswerEvaluation;
  metadata?: Record<string, any>;
}

export interface AnswerEvaluation {
  is_sufficient: boolean;
  score: number;
  missing_elements: string[];
  strengths: string[];
  suggested_follow_up: string | null;
  reasoning: string;
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

export interface WSMessage {
  type: WSMessageType;
  content?: string;
  timestamp?: string;
  data?: Record<string, any>;
  error?: string;
  audio_base64?: string;
}

export interface WSClientMessage {
  type: 'start' | 'answer' | 'answer_audio' | 'skip' | 'end' | 'ping';
  content?: string;
  data?: Record<string, any>;
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