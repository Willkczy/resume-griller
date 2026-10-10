'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { Send, SkipForward, StopCircle, Loader2, Wifi, WifiOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { ChatMessage } from './ChatMessage';
import { VoiceRecorder } from './VoiceRecorder';
import { EvaluationNotice } from './EvaluationNotice';
import { InterviewWebSocket } from '@/lib/websocket';
import { getVoiceStatus } from '@/lib/api';
import { canAutomaticallyStart, commandErrorMessage, isInterviewSnapshot, reconcileAnswerDraft } from '@/lib/interview-outcomes';
import type { PendingAnswer } from '@/lib/interview-outcomes';
import type { AnswerEvaluation, ConversationMessage, GenerationFailure, InterviewAction, SessionStatus, WSMessage } from '@/types';
import { cn } from '@/lib/utils';

interface InterviewRoomProps { sessionId: string; }
interface Message extends ConversationMessage { audioBase64?: string; }

export function InterviewRoom({ sessionId }: InterviewRoomProps) {
  const router = useRouter();
  const [inputValue, setInputValue] = useState('');
  const [canRecord, setCanRecord] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<SessionStatus>('pending');
  const [currentQuestionId, setCurrentQuestionId] = useState<string | null>(null);
  const [questionNumber, setQuestionNumber] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evaluation, setEvaluation] = useState<AnswerEvaluation | null>(null);
  const [generationFailure, setGenerationFailure] = useState<GenerationFailure | null>(null);
  const [allowedActions, setAllowedActions] = useState<InterviewAction[]>([]);
  const [hasRetainedAudio, setHasRetainedAudio] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<InterviewWebSocket | null>(null);
  const draftRef = useRef('');
  const currentQuestionIdRef = useRef<string | null>(null);
  const pendingAnswerRef = useRef<PendingAnswer | null>(null);
  const pendingAudioRef = useRef<string | null>(null);
  const hasAttemptedStartRef = useRef(false);

  useEffect(() => {
    let active = true;
    getVoiceStatus().then((voice) => {
      if (active) setCanRecord(voice.enabled && voice.stt_available);
    }).catch(() => { if (active) setCanRecord(false); });
    return () => { active = false; };
  }, []);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, evaluation, generationFailure]);

  const playAudio = useCallback((base64: string) => {
    setIsSpeaking(true);
    const audio = new Audio(`data:audio/mp3;base64,${base64}`);
    audio.onended = () => setIsSpeaking(false);
    audio.onerror = () => setIsSpeaking(false);
    audio.play().catch(() => setIsSpeaking(false));
  }, []);

  useEffect(() => {
    const websocket = new InterviewWebSocket(sessionId);
    wsRef.current = websocket;
    const unsubMessage = websocket.onMessage((message: WSMessage) => {
      if (message.type === 'pong') return;
      if (message.type === 'transcript') {
        const text = message.content ?? '';
        draftRef.current = text;
        setInputValue(text);
        pendingAnswerRef.current = { questionId: currentQuestionIdRef.current, text };
        return;
      }

      if (message.type === 'connected') {
        setIsConnected(true);
        setIsReconnecting(false);
      }
      if (isInterviewSnapshot(message.data)) {
        const snapshot = message.data;
        const draft = reconcileAnswerDraft(snapshot, draftRef.current, currentQuestionIdRef.current, pendingAnswerRef.current);
        draftRef.current = draft;
        setInputValue(draft);
        const questionChanged = currentQuestionIdRef.current !== snapshot.current_question_id;
        if (snapshot.draft_answer !== null || snapshot.answer_status === 'accepted' || questionChanged || snapshot.status === 'completed' || snapshot.status === 'cancelled') {
          pendingAudioRef.current = null;
          setHasRetainedAudio(false);
        }
        pendingAnswerRef.current = null;
        currentQuestionIdRef.current = snapshot.current_question_id;
        setCurrentQuestionId(snapshot.current_question_id);
        setStatus(snapshot.status);
        setQuestionNumber(snapshot.question_number ?? (snapshot.current_question ? (snapshot.current_question_index ?? 0) + 1 : 0));
        setTotalQuestions(snapshot.total_questions);
        setEvaluation(snapshot.evaluation);
        setGenerationFailure(snapshot.generation_failure);
        setAllowedActions(snapshot.allowed_actions);
        setMessages(snapshot.conversation.map((entry) => ({ ...entry,
          audioBase64: entry.role === 'interviewer' && entry.question_id === snapshot.current_question_id ? message.audio_base64 : undefined,
        })));
        setError(commandErrorMessage(message));
        setIsLoading(false);
        if (message.type === 'connected' && canAutomaticallyStart(snapshot, hasAttemptedStartRef.current)) {
          hasAttemptedStartRef.current = true;
          setIsLoading(true);
          websocket.startInterview(snapshot.total_questions, false);
        }
      } else if (message.type === 'error') {
        // Validation/voice errors without a checkpoint must not discard local drafts or feedback.
        setError(message.error || 'The command failed. Your answer is retained.');
        setIsLoading(false);
      }
      if (message.type === 'question' || message.type === 'follow_up') {
        if (message.audio_base64) playAudio(message.audio_base64);
      }
      if (message.type === 'complete') {
        setIsLoading(false);
      }
    });
    const unsubDisconnect = websocket.onDisconnect(() => {
      setIsConnected(false);
      setIsLoading(false);
      setIsReconnecting(false);
    });
    websocket.connect().catch(() => {
      setError('Could not connect. Reconnect to check the interview state.');
      setIsLoading(false);
    });
    const pingInterval = setInterval(() => { if (websocket.isConnected) websocket.ping(); }, 30000);
    return () => {
      clearInterval(pingInterval);
      unsubMessage();
      unsubDisconnect();
      websocket.disconnect();
      wsRef.current = null;
    };
  }, [sessionId, playAudio]);

  const canAct = (action: InterviewAction) => isConnected && !isLoading && allowedActions.includes(action);
  const canSend = (action: InterviewAction) => canAct(action) && !!wsRef.current?.isConnected;
  const handleStart = () => {
    if (!canSend('start')) return;
    hasAttemptedStartRef.current = true;
    setError(null);
    setIsLoading(true);
    wsRef.current!.startInterview(totalQuestions, false);
  };
  const handleReconnect = () => {
    if (isConnected || isReconnecting || !wsRef.current) return;
    setIsReconnecting(true);
    setError(null);
    wsRef.current.connect().catch(() => {
      setIsReconnecting(false);
      setError('Could not reconnect. Your local answer is still retained.');
    });
  };
  const handleSendText = () => {
    if (!inputValue.trim() || !canSend('answer')) return;
    pendingAnswerRef.current = { questionId: currentQuestionIdRef.current, text: inputValue };
    setError(null);
    setIsLoading(true);
    wsRef.current!.sendAnswer(inputValue, false);
  };
  const handleSendAudio = (audioBase64: string) => {
    pendingAudioRef.current = audioBase64;
    setHasRetainedAudio(true);
    if (!canSend('answer')) return;
    setError(null);
    setIsLoading(true);
    wsRef.current!.sendAudioAnswer(audioBase64, false);
  };
  const handleSkip = () => {
    if (!canSend('skip')) return;
    setError(null);
    setIsLoading(true);
    wsRef.current!.skipQuestion(false);
  };
  const handleEnd = () => {
    if (!canSend('end')) return;
    if (confirm('Are you sure you want to end the interview?')) {
      setIsLoading(true);
      wsRef.current!.endInterview();
    }
  };
  const progress = totalQuestions > 0 ? (questionNumber / totalQuestions) * 100 : 0;
  const terminal = status === 'completed' || status === 'cancelled';
  const evaluatedQuestion = evaluation?.question_id ? messages.find((message) => message.role === 'interviewer' && message.question_id === evaluation.question_id)?.content : null;

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] text-gray-900">
      <div className="flex-shrink-0 border-b bg-white p-4">
        <div className="flex items-center justify-between mb-3 gap-2">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">{status === 'pending' ? 'Interview pending' : terminal ? `Interview ${status}` : `Question ${questionNumber} of ${totalQuestions}`}</span>
            <span className={cn('flex items-center gap-1 text-xs px-2 py-1 rounded-full', isConnected ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700')}>
              {isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}{isConnected ? 'Connected' : 'Disconnected'}
            </span>
            {!isConnected && <Button variant="outline" size="sm" onClick={handleReconnect} disabled={isReconnecting}>{isReconnecting ? 'Reconnecting...' : 'Reconnect'}</Button>}
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleSkip} disabled={!canAct('skip')}><SkipForward className="w-4 h-4 mr-1" />Skip</Button>
            <Button variant="destructive" size="sm" onClick={handleEnd} disabled={!canAct('end')}><StopCircle className="w-4 h-4 mr-1" />End</Button>
          </div>
        </div>
        <Progress value={progress} className="h-2" />
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.map((message, index) => <ChatMessage key={`${message.timestamp}-${message.answer_id ?? message.question_id ?? index}-${index}`} role={message.role} content={message.content} timestamp={new Date(message.timestamp)} isFollowUp={message.is_follow_up} audioBase64={message.audioBase64} />)}
        <EvaluationNotice evaluation={evaluation} generationFailure={generationFailure} canAnswer={allowedActions.includes('answer')} evaluatedQuestion={evaluatedQuestion} isPreviousAnswer={!!evaluation?.question_id && evaluation.question_id !== currentQuestionId} />
        {status === 'pending' && !isLoading && <div className="text-center"><Button onClick={handleStart} disabled={!canAct('start')}>{generationFailure ? 'Try again' : 'Start interview'}</Button></div>}
        {isLoading && <div className="flex items-center justify-center gap-2 text-gray-500"><Loader2 className="w-4 h-4 animate-spin" /><span className="text-sm">{status === 'pending' ? 'Starting interview...' : 'Processing your answer...'}</span></div>}
        <div ref={messagesEndRef} />
      </div>
      {error && <div className="flex-shrink-0 bg-red-50 border-t border-red-200 p-3" role="alert"><p className="text-sm text-red-600">{error}</p></div>}
      {!terminal && status === 'in_progress' && (
        <div className="flex-shrink-0 border-t bg-white p-4">
          <div className="flex items-center gap-3">
            <Input value={inputValue} onChange={(event) => { draftRef.current = event.target.value; setInputValue(event.target.value); }} onKeyDown={(event) => { if (event.key === 'Enter') handleSendText(); }} placeholder="Type your answer..." disabled={!canAct('answer')} aria-label="Your answer" />
            <Button onClick={handleSendText} disabled={!inputValue.trim() || !canAct('answer')} aria-label="Submit answer"><Send className="w-4 h-4" /></Button>
            {canRecord && <VoiceRecorder onRecordingComplete={handleSendAudio} disabled={!canAct('answer') || isSpeaking} />}
          </div>
          {hasRetainedAudio && !inputValue && <div className="mt-2 text-sm"><p>Recorded audio is retained in this page. Reconnect to check whether it was accepted before retrying.</p><Button variant="outline" size="sm" onClick={() => { if (pendingAudioRef.current) handleSendAudio(pendingAudioRef.current); }} disabled={!canAct('answer')}>Submit retained recording</Button></div>}
          <p className="text-xs text-gray-500 mt-2 text-center">{!isConnected ? 'Your local answer is retained. Reconnect to check the server state; it will not be resubmitted automatically.' : isLoading ? 'Your answer stays here until the server confirms the result.' : !allowedActions.includes('answer') ? 'Choose an available action above to continue.' : canRecord ? 'Type your answer or hold the microphone button to speak.' : 'Type your answer to continue.'}</p>
        </div>
      )}
      {terminal && <div className="flex-shrink-0 border-t bg-green-50 p-4 text-center"><p className="text-green-700 font-medium">Interview {status}.</p><Button className="mt-2" onClick={() => router.push(`/result/${sessionId}`)}>View results</Button></div>}
    </div>
  );
}
