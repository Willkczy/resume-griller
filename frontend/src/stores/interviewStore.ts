// frontend/src/stores/interviewStore.ts

import { create } from 'zustand';
import type { 
  ConversationMessage, 
  InterviewMode, 
  SessionStatus,
  AnswerEvaluation 
} from '@/types';

interface Message {
  id: string;
  role: 'interviewer' | 'candidate' | 'system';
  content: string;
  timestamp: Date;
  isFollowUp?: boolean;
  evaluation?: AnswerEvaluation;
  audioBase64?: string;
}

interface InterviewState {
  // Session info
  sessionId: string | null;
  resumeId: string | null;
  mode: InterviewMode;
  status: SessionStatus;
  
  // Interview progress
  currentQuestion: string | null;
  questionNumber: number;
  totalQuestions: number;
  
  // Messages
  messages: Message[];
  
  // Voice
  isVoiceEnabled: boolean;
  isRecording: boolean;
  isSpeaking: boolean;
  
  // Connection
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  setSession: (sessionId: string, resumeId: string, mode: InterviewMode) => void;
  setStatus: (status: SessionStatus) => void;
  setQuestion: (question: string, questionNumber: number, totalQuestions: number) => void;
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void;
  setVoiceEnabled: (enabled: boolean) => void;
  setRecording: (recording: boolean) => void;
  setSpeaking: (speaking: boolean) => void;
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialState = {
  sessionId: null,
  resumeId: null,
  mode: 'mixed' as InterviewMode,
  status: 'pending' as SessionStatus,
  currentQuestion: null,
  questionNumber: 0,
  totalQuestions: 0,
  messages: [],
  isVoiceEnabled: true,
  isRecording: false,
  isSpeaking: false,
  isConnected: false,
  isLoading: false,
  error: null,
};

export const useInterviewStore = create<InterviewState>((set) => ({
  ...initialState,

  setSession: (sessionId, resumeId, mode) => 
    set({ sessionId, resumeId, mode, status: 'pending' }),

  setStatus: (status) => set({ status }),

  setQuestion: (question, questionNumber, totalQuestions) =>
    set({ currentQuestion: question, questionNumber, totalQuestions }),

  addMessage: (message) =>
    set((state) => ({
      messages: [
        ...state.messages,
        {
          ...message,
          id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          timestamp: new Date(),
        },
      ],
    })),

  setVoiceEnabled: (enabled) => set({ isVoiceEnabled: enabled }),
  setRecording: (recording) => set({ isRecording: recording }),
  setSpeaking: (speaking) => set({ isSpeaking: speaking }),
  setConnected: (connected) => set({ isConnected: connected }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),

  reset: () => set(initialState),
}));