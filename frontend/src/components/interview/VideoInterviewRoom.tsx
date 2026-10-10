'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { Mic, MicOff, Volume2, VolumeX, SkipForward, StopCircle, Loader2, Wifi, WifiOff, Video, VideoOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { EvaluationNotice } from './EvaluationNotice';
import { InterviewWebSocket } from '@/lib/websocket';
import { getVoiceStatus } from '@/lib/api';
import { acceptedAnswerForQuestion, canAutomaticallyStart, commandErrorMessage, isInterviewSnapshot, reconcileAnswerDraft } from '@/lib/interview-outcomes';
import type { PendingAnswer } from '@/lib/interview-outcomes';
import type { AnswerEvaluation, ConversationMessage, GenerationFailure, InterviewAction, SessionStatus, WSMessage } from '@/types';
import { cn } from '@/lib/utils';

interface VideoInterviewRoomProps { sessionId: string; }
type InterviewPhase = 'waiting' | 'question' | 'answering' | 'processing';

export function VideoInterviewRoom({ sessionId }: VideoInterviewRoomProps) {
  const router = useRouter();
  const [status, setStatus] = useState<SessionStatus>('pending');
  const [currentQuestionId, setCurrentQuestionId] = useState<string | null>(null);
  const [phase, setPhase] = useState<InterviewPhase>('waiting');
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [questionNumber, setQuestionNumber] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evaluation, setEvaluation] = useState<AnswerEvaluation | null>(null);
  const [generationFailure, setGenerationFailure] = useState<GenerationFailure | null>(null);
  const [allowedActions, setAllowedActions] = useState<InterviewAction[]>([]);
  const [conversation, setConversation] = useState<ConversationMessage[]>([]);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const [canRecord, setCanRecord] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [acceptedTranscript, setAcceptedTranscript] = useState('');
  const [hasRetainedAudio, setHasRetainedAudio] = useState(false);
  const [hasQuestionAudio, setHasQuestionAudio] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const wsRef = useRef<InterviewWebSocket | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const voiceEnabledRef = useRef(voiceEnabled);
  const draftRef = useRef('');
  const currentQuestionIdRef = useRef<string | null>(null);
  const pendingAnswerRef = useRef<PendingAnswer | null>(null);
  const pendingAudioRef = useRef<string | null>(null);
  const questionAudioRef = useRef<string | null>(null);
  const allowedActionsRef = useRef<InterviewAction[]>([]);
  const hasAttemptedStartRef = useRef(false);
  useEffect(() => { voiceEnabledRef.current = voiceEnabled; }, [voiceEnabled]);

  useEffect(() => {
    let active = true;
    getVoiceStatus().then((voice) => { if (active) setCanRecord(voice.enabled && voice.stt_available); }).catch(() => { if (active) setCanRecord(false); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    async function initCamera() {
      if (!cameraEnabled) { if (videoRef.current) videoRef.current.srcObject = null; return; }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (!active) { stream.getTracks().forEach((track) => track.stop()); return; }
        mediaStreamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch { if (active) setCameraEnabled(false); }
    }
    initCamera();
    return () => { active = false; mediaStreamRef.current?.getTracks().forEach((track) => track.stop()); mediaStreamRef.current = null; };
  }, [cameraEnabled]);

  const playAudio = useCallback((base64: string) => {
    audioRef.current?.pause();
    setIsSpeaking(true);
    const audio = new Audio(`data:audio/mp3;base64,${base64}`);
    audioRef.current = audio;
    audio.onended = () => setIsSpeaking(false);
    audio.onerror = () => setIsSpeaking(false);
    audio.play().catch(() => setIsSpeaking(false));
  }, []);
  const stopAudio = useCallback(() => { audioRef.current?.pause(); audioRef.current = null; setIsSpeaking(false); }, []);

  useEffect(() => {
    const websocket = new InterviewWebSocket(sessionId);
    wsRef.current = websocket;
    const unsubMessage = websocket.onMessage((message: WSMessage) => {
      if (message.type === 'pong') return;
      if (message.type === 'transcript') {
        const text = message.content ?? '';
        draftRef.current = text;
        setTranscript(text);
        pendingAnswerRef.current = { questionId: currentQuestionIdRef.current, text };
        return;
      }
      if (message.type === 'connected') { setIsConnected(true); setIsReconnecting(false); }
      if (isInterviewSnapshot(message.data)) {
        const snapshot = message.data;
        const draft = reconcileAnswerDraft(snapshot, draftRef.current, currentQuestionIdRef.current, pendingAnswerRef.current);
        draftRef.current = draft;
        setTranscript(draft);
        const questionChanged = currentQuestionIdRef.current !== snapshot.current_question_id;
        if (snapshot.draft_answer !== null || snapshot.answer_status === 'accepted' || questionChanged || snapshot.status === 'completed' || snapshot.status === 'cancelled') {
          pendingAudioRef.current = null;
          setHasRetainedAudio(false);
        }
        if (questionChanged) { questionAudioRef.current = null; setHasQuestionAudio(false); }
        pendingAnswerRef.current = null;
        currentQuestionIdRef.current = snapshot.current_question_id;
        setCurrentQuestionId(snapshot.current_question_id);
        allowedActionsRef.current = snapshot.allowed_actions;
        setAllowedActions(snapshot.allowed_actions);
        setStatus(snapshot.status);
        setCurrentQuestion(snapshot.current_question ?? '');
        setQuestionNumber(snapshot.question_number ?? (snapshot.current_question ? (snapshot.current_question_index ?? 0) + 1 : 0));
        setTotalQuestions(snapshot.total_questions);
        setEvaluation(snapshot.evaluation);
        setGenerationFailure(snapshot.generation_failure);
        setConversation(snapshot.conversation);
        setAcceptedTranscript(acceptedAnswerForQuestion(snapshot.conversation, snapshot.current_question_id));
        setError(commandErrorMessage(message));
        setPhase('question');
        if (message.type === 'connected' && canAutomaticallyStart(snapshot, hasAttemptedStartRef.current)) {
          hasAttemptedStartRef.current = true;
          setPhase('processing');
          websocket.startInterview(snapshot.total_questions, voiceEnabledRef.current);
        }
      } else if (message.type === 'error') {
        setError(message.error || 'The command failed. Your answer is retained.');
        setPhase('question');
      }
      if ((message.type === 'question' || message.type === 'follow_up') && message.audio_base64) {
        questionAudioRef.current = message.audio_base64;
        setHasQuestionAudio(true);
        if (voiceEnabledRef.current) playAudio(message.audio_base64);
      }
      if (message.type === 'complete') setPhase('question');
    });
    const unsubDisconnect = websocket.onDisconnect(() => { setIsConnected(false); setIsReconnecting(false); setPhase('question'); });
    websocket.connect().catch(() => { setError('Could not connect. Reconnect to check the interview state.'); setPhase('question'); });
    const pingInterval = setInterval(() => { if (websocket.isConnected) websocket.ping(); }, 30000);
    return () => {
      clearInterval(pingInterval);
      unsubMessage(); unsubDisconnect(); websocket.disconnect(); wsRef.current = null;
      audioRef.current?.pause();
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== 'inactive') { recorder.onstop = null; recorder.stop(); recorder.stream.getTracks().forEach((track) => track.stop()); }
    };
  }, [sessionId, playAudio]);

  const isBusy = phase === 'waiting' || phase === 'processing' || phase === 'answering';
  const canAct = (action: InterviewAction) => isConnected && !isBusy && !isRecording && allowedActions.includes(action);
  const canSend = (action: InterviewAction) => canAct(action) && !!wsRef.current?.isConnected;
  const handleStart = () => {
    if (!canSend('start')) return;
    hasAttemptedStartRef.current = true; setPhase('processing'); setError(null);
    wsRef.current!.startInterview(totalQuestions, voiceEnabledRef.current);
  };
  const handleReconnect = () => {
    if (isConnected || isReconnecting || !wsRef.current) return;
    setIsReconnecting(true); setError(null);
    wsRef.current.connect().catch(() => { setIsReconnecting(false); setError('Could not reconnect. Your local answer is still retained.'); });
  };
  const submitAudio = (audioBase64: string) => {
    if (!canSend('answer')) return;
    setError(null); setPhase('processing');
    wsRef.current!.sendAudioAnswer(audioBase64, voiceEnabledRef.current);
  };
  const submitTranscript = () => {
    if (!transcript.trim() || !canSend('answer')) return;
    pendingAnswerRef.current = { questionId: currentQuestionIdRef.current, text: transcript };
    setError(null); setPhase('processing');
    wsRef.current!.sendAnswer(transcript, voiceEnabledRef.current);
  };
  const startRecording = async () => {
    if (!canRecord || !canSend('answer') || isSpeaking) return;
    const recordingQuestionId = currentQuestionIdRef.current;
    setPhase('answering');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!wsRef.current?.isConnected || recordingQuestionId !== currentQuestionIdRef.current || !allowedActionsRef.current.includes('answer')) {
        stream.getTracks().forEach((track) => track.stop()); setPhase('question'); return;
      }
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = recorder; audioChunksRef.current = [];
      recorder.ondataavailable = (event) => { if (event.data.size > 0) audioChunksRef.current.push(event.data); };
      recorder.onstop = () => {
        setIsRecording(false);
        const reader = new FileReader();
        reader.onloadend = () => {
          const audioBase64 = (reader.result as string).split(',')[1];
          pendingAudioRef.current = audioBase64; setHasRetainedAudio(true);
          if (wsRef.current?.isConnected && recordingQuestionId === currentQuestionIdRef.current && allowedActionsRef.current.includes('answer')) {
            setPhase('processing'); wsRef.current.sendAudioAnswer(audioBase64, voiceEnabledRef.current);
          } else { setPhase('question'); }
        };
        reader.readAsDataURL(new Blob(audioChunksRef.current, { type: 'audio/webm' }));
        stream.getTracks().forEach((track) => track.stop());
      };
      recorder.start(); setIsRecording(true);
    } catch { setPhase('question'); setError('Could not access the microphone. You can type your answer below.'); }
  };
  const stopRecording = () => { if (mediaRecorderRef.current?.state === 'recording') mediaRecorderRef.current.stop(); };
  const handleSkip = () => { if (!canSend('skip')) return; stopAudio(); setPhase('processing'); setError(null); wsRef.current!.skipQuestion(voiceEnabledRef.current); };
  const handleEnd = () => {
    if (!canSend('end')) return;
    if (confirm('Are you sure you want to end the interview?')) { stopAudio(); setPhase('processing'); wsRef.current!.endInterview(); }
  };
  const progress = totalQuestions > 0 ? (questionNumber / totalQuestions) * 100 : 0;
  const terminal = status === 'completed' || status === 'cancelled';
  const evaluatedQuestion = evaluation?.question_id ? conversation.find((message) => message.role === 'interviewer' && message.question_id === evaluation.question_id)?.content : null;

  return (
    <div className="flex flex-col min-h-screen bg-gray-900 text-white">
      <div className="flex-shrink-0 flex flex-wrap items-center justify-between gap-3 px-6 py-3 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-3"><h1 className="text-lg font-semibold">Resume Griller</h1><span className={cn('flex items-center gap-1 text-xs px-2 py-1 rounded-full', isConnected ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300')}>{isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}{isConnected ? 'Connected' : 'Disconnected'}</span>{!isConnected && <Button size="sm" onClick={handleReconnect} disabled={isReconnecting}>{isReconnecting ? 'Reconnecting...' : 'Reconnect'}</Button>}</div>
        <div className="flex items-center gap-4"><span className="text-sm text-gray-400">{status === 'pending' ? 'Interview pending' : terminal ? `Interview ${status}` : `Question ${questionNumber} of ${totalQuestions}`}</span><Button variant="ghost" size="sm" onClick={() => setVoiceEnabled(!voiceEnabled)} aria-label={voiceEnabled ? 'Mute question audio' : 'Enable question audio'}>{voiceEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}</Button><Button variant="ghost" size="sm" onClick={() => setCameraEnabled(!cameraEnabled)} aria-label={cameraEnabled ? 'Turn camera off' : 'Turn camera on'}>{cameraEnabled ? <Video className="w-4 h-4" /> : <VideoOff className="w-4 h-4" />}</Button><Button variant="destructive" size="sm" onClick={handleEnd} disabled={!canAct('end')}><StopCircle className="w-4 h-4 mr-1" />End</Button></div>
      </div>
      <div className="flex-1 flex flex-col items-center p-6 gap-6">
        <div className="relative w-full max-w-xl aspect-video bg-gray-800 rounded-2xl overflow-hidden shadow-2xl">
          {cameraEnabled ? <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover" style={{ transform: 'scaleX(-1)' }} /> : <div className="w-full h-full flex items-center justify-center text-gray-500"><VideoOff className="w-10 h-10 mr-2" />Camera is off</div>}
          {isRecording && <div className="absolute top-4 left-4 bg-red-600 px-3 py-1 rounded-full text-sm">Recording</div>}
          {isSpeaking && <div className="absolute top-4 right-4 bg-blue-600 px-3 py-1 rounded-full text-sm">Speaking</div>}
        </div>
        <div className="w-full max-w-3xl space-y-4">
          {terminal ? <div className="text-center text-green-400 text-xl">Interview {status}.<div className="mt-3"><Button onClick={() => router.push(`/result/${sessionId}`)}>View results</Button></div></div> : currentQuestion && <div className="bg-gray-800 rounded-xl p-6"><p className="text-xl md:text-2xl font-medium text-center leading-relaxed">{currentQuestion}</p>{acceptedTranscript && <p className="text-sm text-gray-300 mt-4 border-t border-gray-700 pt-4">Accepted answer: {acceptedTranscript}</p>}</div>}
          <EvaluationNotice evaluation={evaluation} generationFailure={generationFailure} canAnswer={allowedActions.includes('answer')} evaluatedQuestion={evaluatedQuestion} isPreviousAnswer={!!evaluation?.question_id && evaluation.question_id !== currentQuestionId} dark />
          {status === 'pending' && !isBusy && <div className="text-center"><Button onClick={handleStart} disabled={!canAct('start')}>{generationFailure ? 'Try again' : 'Start interview'}</Button></div>}
          {isBusy && !isRecording && <div className="flex items-center justify-center gap-2 text-gray-400"><Loader2 className="w-5 h-5 animate-spin" />{status === 'pending' ? 'Preparing your interview...' : 'Processing your answer...'}</div>}
          {error && <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 text-red-300" role="alert">{error}</div>}
          {conversation.length > 0 && <details className="rounded-lg border border-gray-700 p-3"><summary className="cursor-pointer text-sm text-gray-300">Accepted conversation ({conversation.length} messages)</summary><ol className="mt-3 space-y-3 text-sm">{conversation.map((message, index) => <li key={`${message.timestamp}-${index}`}><span className="font-medium capitalize">{message.role}: </span><span className="whitespace-pre-wrap">{message.content}</span></li>)}</ol></details>}
        </div>
      </div>
      {!terminal && status === 'in_progress' && <div className="flex-shrink-0 bg-gray-800 border-t border-gray-700 px-6 py-4"><div className="max-w-3xl mx-auto space-y-3">
        <Progress value={progress} className="h-1" />
        <label className="block text-sm text-gray-300" htmlFor="answer-transcript">Your answer / transcript</label>
        <textarea id="answer-transcript" className="w-full rounded-md border border-gray-600 bg-gray-900 p-3 text-white disabled:opacity-60" rows={3} value={transcript} onChange={(event) => { draftRef.current = event.target.value; setTranscript(event.target.value); }} disabled={!canAct('answer')} placeholder="Speak to transcribe, or type your answer here." />
        <div className="flex flex-wrap items-center justify-center gap-3">
          {hasQuestionAudio && <Button variant="outline" onClick={() => { if (questionAudioRef.current) playAudio(questionAudioRef.current); }} disabled={!isConnected || isSpeaking || isBusy || isRecording}><Volume2 className="w-4 h-4 mr-2" />Read question</Button>}
          {isRecording ? <Button onClick={stopRecording} className="bg-red-600 hover:bg-red-700"><MicOff className="w-5 h-5 mr-2" />Stop recording</Button> : canRecord && <Button onClick={startRecording} disabled={!canAct('answer') || isSpeaking} className="bg-green-600 hover:bg-green-700"><Mic className="w-5 h-5 mr-2" />Start answering</Button>}
          <Button onClick={submitTranscript} disabled={!transcript.trim() || !canAct('answer')}>Submit answer</Button>
          <Button variant="outline" onClick={handleSkip} disabled={!canAct('skip')}><SkipForward className="w-4 h-4 mr-2" />Skip</Button>
          {hasRetainedAudio && !transcript && <Button variant="outline" onClick={() => { if (pendingAudioRef.current) submitAudio(pendingAudioRef.current); }} disabled={!canAct('answer')}>Submit retained recording</Button>}
        </div>
        <p className="text-xs text-gray-400 text-center">{!isConnected ? 'Your local answer is retained. Reconnect to check the server state; it will not be resubmitted automatically.' : isRecording ? 'Stop recording to submit your answer.' : isBusy ? 'Your transcript stays here until the server confirms the result.' : !allowedActions.includes('answer') ? 'Choose an available action to continue.' : 'Submit a typed answer or record your response.'}</p>
      </div></div>}
    </div>
  );
}
