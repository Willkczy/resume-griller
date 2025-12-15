// frontend/src/lib/api.ts

import type {
  ResumeUploadResponse,
  ResumeSummary,
  SessionCreateRequest,
  SessionResponse,
  SessionDetail,
  SessionSummary,
  InterviewResponse,
  VoiceStatus,
} from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const response = await fetch(url, {
    ...options,
    headers: {
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, error.detail || 'Request failed');
  }

  return response.json();
}

// ============== Health ==============

export async function checkHealth(): Promise<{ status: string }> {
  return fetchApi('/health');
}

// ============== Resume APIs ==============

export async function uploadResume(file: File): Promise<ResumeUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/v1/resume/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new ApiError(response.status, error.detail);
  }

  return response.json();
}

export async function getResumeSummary(resumeId: string): Promise<ResumeSummary> {
  return fetchApi(`/api/v1/resume/${resumeId}`);
}

export async function deleteResume(resumeId: string): Promise<void> {
  await fetchApi(`/api/v1/resume/${resumeId}`, { method: 'DELETE' });
}

// ============== Session APIs ==============

export async function createSession(
  request: SessionCreateRequest
): Promise<InterviewResponse & { metadata: { session_id: string } }> {
  return fetchApi('/api/v1/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

export async function listSessions(resumeId?: string): Promise<SessionResponse[]> {
  const query = resumeId ? `?resume_id=${resumeId}` : '';
  return fetchApi(`/api/v1/sessions${query}`);
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  return fetchApi(`/api/v1/sessions/${sessionId}`);
}

export async function submitAnswer(
  sessionId: string,
  answer: string
): Promise<InterviewResponse> {
  return fetchApi(`/api/v1/sessions/${sessionId}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answer }),
  });
}

export async function skipQuestion(sessionId: string): Promise<InterviewResponse> {
  return fetchApi(`/api/v1/sessions/${sessionId}/skip`, { method: 'POST' });
}

export async function endSession(sessionId: string): Promise<InterviewResponse> {
  return fetchApi(`/api/v1/sessions/${sessionId}/end`, { method: 'POST' });
}

export async function getSessionSummary(sessionId: string): Promise<SessionSummary> {
  return fetchApi(`/api/v1/sessions/${sessionId}/summary`);
}

// ============== Voice APIs ==============

export async function getVoiceStatus(): Promise<VoiceStatus> {
  return fetchApi('/api/v1/voice/status');
}

export async function synthesizeSpeech(text: string): Promise<{ audio_base64: string }> {
  return fetchApi('/api/v1/voice/synthesize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
}

export { ApiError };