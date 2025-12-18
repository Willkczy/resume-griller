// frontend/src/components/interview/InterviewRoom.tsx

'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { Send, SkipForward, StopCircle, Loader2, Wifi, WifiOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { ChatMessage } from './ChatMessage';
import { VoiceRecorder } from './VoiceRecorder';
import { InterviewWebSocket } from '@/lib/websocket';
import type { WSMessage } from '@/types';
import { cn } from '@/lib/utils';

interface InterviewRoomProps {
  sessionId: string;
}

interface Message {
  id: string;
  role: 'interviewer' | 'candidate' | 'system';
  content: string;
  timestamp: Date;
  isFollowUp?: boolean;
  audioBase64?: string;
}

type SessionStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled';

export function InterviewRoom({ sessionId }: InterviewRoomProps) {
  const router = useRouter();
  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<SessionStatus>('pending');
  const [questionNumber, setQuestionNumber] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<InterviewWebSocket | null>(null);
  const isInitializedRef = useRef(false);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Add message helper
  const addMessage = useCallback((msg: Omit<Message, 'id' | 'timestamp'>) => {
    const newMessage = {
      ...msg,
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newMessage]);
    return newMessage;
  }, []);

  // Play audio helper
  const playAudio = useCallback((base64: string) => {
    setIsSpeaking(true);
    const audio = new Audio(`data:audio/mp3;base64,${base64}`);
    audio.onended = () => setIsSpeaking(false);
    audio.onerror = () => setIsSpeaking(false);
    audio.play().catch(() => setIsSpeaking(false));
  }, []);

  // Initialize WebSocket
  useEffect(() => {
    // Prevent double initialization in React Strict Mode
    if (isInitializedRef.current) {
      return;
    }
    isInitializedRef.current = true;

    console.log('[InterviewRoom] Initializing WebSocket for session:', sessionId);

    const websocket = new InterviewWebSocket(sessionId);
    wsRef.current = websocket;

    // Handle messages
    const unsubMessage = websocket.onMessage((message: WSMessage) => {
      console.log('[InterviewRoom] Message received:', message.type, message);

      switch (message.type) {
        case 'connected':
          console.log('[InterviewRoom] Connected, status:', message.data?.status);
          setIsConnected(true);
          setError(null);

          // Handle existing session state
          if (message.data?.status === 'in_progress' && message.data?.current_question) {
            // Session already has a question, show it
            setStatus('in_progress');
            setQuestionNumber(message.data.question_number || 1);
            setTotalQuestions(message.data.total_questions || 5);

            // Add the current question to messages
            setMessages((prev) => {
              // Avoid duplicates
              if (prev.some((m) => m.content === message.data?.current_question)) {
                return prev;
              }
              return [
                ...prev,
                {
                  id: `msg-${Date.now()}`,
                  role: 'interviewer' as const,
                  content: message.data!.current_question,
                  timestamp: new Date(),
                  isFollowUp: false,
                },
              ];
            });
            setIsLoading(false);
          } else if (message.data?.status === 'pending') {
            // Need to start the interview
            console.log('[InterviewRoom] Starting interview...');
            websocket.startInterview(message.data?.total_questions || 5);
          } else {
            setIsLoading(false);
          }
          break;

        case 'question':
        case 'follow_up':
          console.log('[InterviewRoom] Received question/follow_up');
          setStatus('in_progress');
          setQuestionNumber(message.data?.question_number || questionNumber);
          setTotalQuestions(message.data?.total_questions || totalQuestions);

          addMessage({
            role: 'interviewer',
            content: message.content || '',
            isFollowUp: message.type === 'follow_up',
            audioBase64: message.audio_base64,
          });

          // Auto-play audio
          if (message.audio_base64) {
            playAudio(message.audio_base64);
          }
          setIsLoading(false);
          break;

        case 'transcript':
          // Show what the AI heard
          addMessage({
            role: 'system',
            content: `🎤 Heard: "${message.content}"`,
          });
          break;

        case 'complete':
          setStatus('completed');
          addMessage({
            role: 'system',
            content: message.content || 'Interview completed!',
          });
          setIsLoading(false);
          // Redirect to results after a delay
          setTimeout(() => {
            router.push(`/result/${sessionId}`);
          }, 2000);
          break;

        case 'error':
          console.error('[InterviewRoom] Error:', message.error);
          setError(message.error || 'An error occurred');
          setIsLoading(false);
          break;

        case 'pong':
          // Keep-alive response, ignore
          break;

        default:
          console.log('[InterviewRoom] Unknown message type:', message.type);
      }
    });

    const unsubConnect = websocket.onConnect(() => {
      console.log('[InterviewRoom] WebSocket connected');
    });

    const unsubDisconnect = websocket.onDisconnect(() => {
      console.log('[InterviewRoom] WebSocket disconnected');
      setIsConnected(false);
    });

    const unsubError = websocket.onError((err) => {
      console.error('[InterviewRoom] WebSocket error:', err);
      // Don't set error immediately - might reconnect
    });

    // Connect
    websocket.connect().catch((err) => {
      console.error('[InterviewRoom] Failed to connect:', err);
      setError('Failed to connect to interview. Please refresh the page.');
      setIsLoading(false);
    });

    // Keep-alive ping
    const pingInterval = setInterval(() => {
      if (websocket.isConnected) {
        websocket.ping();
      }
    }, 30000);

    return () => {
      console.log('[InterviewRoom] Cleaning up');
      clearInterval(pingInterval);
      unsubMessage();
      unsubConnect();
      unsubDisconnect();
      unsubError();
      websocket.disconnect();
      isInitializedRef.current = false;
    };
  }, [sessionId, router, addMessage, playAudio]);

  // Send text answer
  const handleSendText = () => {
    if (!inputValue.trim() || !wsRef.current || isLoading) return;

    addMessage({
      role: 'candidate',
      content: inputValue,
    });

    wsRef.current.sendAnswer(inputValue);
    setInputValue('');
    setIsLoading(true);
  };

  // Send audio answer
  const handleSendAudio = (audioBase64: string) => {
    if (!wsRef.current || isLoading) return;

    addMessage({
      role: 'candidate',
      content: '🎤 [Voice message sent]',
    });

    wsRef.current.sendAudioAnswer(audioBase64);
    setIsLoading(true);
  };

  // Skip question
  const handleSkip = () => {
    if (!wsRef.current || isLoading) return;

    addMessage({
      role: 'system',
      content: 'Question skipped',
    });

    wsRef.current.skipQuestion();
    setIsLoading(true);
  };

  // End interview
  const handleEnd = () => {
    if (!wsRef.current) return;

    if (confirm('Are you sure you want to end the interview?')) {
      wsRef.current.endInterview();
      setIsLoading(true);
    }
  };

  const progress = totalQuestions > 0 ? (questionNumber / totalQuestions) * 100 : 0;

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)]">
      {/* Header */}
      <div className="flex-shrink-0 border-b bg-white p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">
              Question {questionNumber} of {totalQuestions}
            </span>
            <span
              className={cn(
                'flex items-center gap-1 text-xs px-2 py-1 rounded-full',
                isConnected ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
              )}
            >
              {isConnected ? (
                <>
                  <Wifi className="w-3 h-3" /> Connected
                </>
              ) : (
                <>
                  <WifiOff className="w-3 h-3" /> Disconnected
                </>
              )}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleSkip}
              disabled={isLoading || status !== 'in_progress'}
            >
              <SkipForward className="w-4 h-4 mr-1" />
              Skip
            </Button>
            <Button variant="destructive" size="sm" onClick={handleEnd} disabled={isLoading}>
              <StopCircle className="w-4 h-4 mr-1" />
              End
            </Button>
          </div>
        </div>

        <Progress value={progress} className="h-2" />
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.length === 0 && isLoading && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <Loader2 className="w-8 h-8 animate-spin mx-auto text-gray-400" />
              <p className="mt-2 text-sm text-gray-500">Starting interview...</p>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <ChatMessage
            key={message.id}
            role={message.role}
            content={message.content}
            timestamp={message.timestamp}
            isFollowUp={message.isFollowUp}
            audioBase64={message.audioBase64}
          />
        ))}

        {isLoading && messages.length > 0 && (
          <div className="flex items-center gap-2 text-gray-500">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">AI is thinking...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Error display */}
      {error && (
        <div className="flex-shrink-0 bg-red-50 border-t border-red-200 p-3">
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      {/* Input Area */}
      {status !== 'completed' && (
        <div className="flex-shrink-0 border-t bg-white p-4">
          <div className="flex items-center gap-3">
            <div className="flex-1">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendText()}
                placeholder="Type your answer..."
                disabled={isLoading || !isConnected}
              />
            </div>

            <Button
              onClick={handleSendText}
              disabled={!inputValue.trim() || isLoading || !isConnected}
            >
              <Send className="w-4 h-4" />
            </Button>

            <VoiceRecorder
              onRecordingComplete={handleSendAudio}
              disabled={isLoading || !isConnected || isSpeaking}
            />
          </div>

          <p className="text-xs text-gray-500 mt-2 text-center">
            Type your answer or hold the microphone button to speak
          </p>
        </div>
      )}

      {/* Completed state */}
      {status === 'completed' && (
        <div className="flex-shrink-0 border-t bg-green-50 p-4 text-center">
          <p className="text-green-700 font-medium">
            🎉 Interview completed! Redirecting to results...
          </p>
        </div>
      )}
    </div>
  );
}