// frontend/src/components/interview/VideoInterviewRoom.tsx

'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import {
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  SkipForward,
  StopCircle,
  Loader2,
  Wifi,
  WifiOff,
  Video,
  VideoOff,
  Settings,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { InterviewWebSocket } from '@/lib/websocket';
import type { WSMessage } from '@/types';
import { cn } from '@/lib/utils';

interface VideoInterviewRoomProps {
  sessionId: string;
}

type SessionStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled';
type InterviewPhase = 'waiting' | 'question' | 'answering' | 'processing';

export function VideoInterviewRoom({ sessionId }: VideoInterviewRoomProps) {
  const router = useRouter();

  // Interview state
  const [status, setStatus] = useState<SessionStatus>('pending');
  const [phase, setPhase] = useState<InterviewPhase>('waiting');
  const [currentQuestion, setCurrentQuestion] = useState<string>('');
  const [questionNumber, setQuestionNumber] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Settings
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [cameraEnabled, setCameraEnabled] = useState(true);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [transcript, setTranscript] = useState<string>('');

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const wsRef = useRef<InterviewWebSocket | null>(null);
  const isInitializedRef = useRef(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Initialize webcam
  useEffect(() => {
    async function initCamera() {
      if (!cameraEnabled) {
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track) => track.stop());
          mediaStreamRef.current = null;
        }
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false, // Audio handled separately for recording
        });
        mediaStreamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('Failed to access camera:', err);
        setCameraEnabled(false);
      }
    }

    initCamera();

    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [cameraEnabled]);

  // Play audio helper
  const playAudio = useCallback((base64: string): Promise<void> => {
    return new Promise((resolve) => {
      setIsSpeaking(true);
      const audio = new Audio(`data:audio/mp3;base64,${base64}`);
      audioRef.current = audio;
      audio.onended = () => {
        setIsSpeaking(false);
        resolve();
      };
      audio.onerror = () => {
        setIsSpeaking(false);
        resolve();
      };
      audio.play().catch(() => {
        setIsSpeaking(false);
        resolve();
      });
    });
  }, []);

  // Stop audio
  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
      setIsSpeaking(false);
    }
  }, []);

  // Initialize WebSocket
  useEffect(() => {
    if (isInitializedRef.current) return;
    isInitializedRef.current = true;

    const websocket = new InterviewWebSocket(sessionId);
    wsRef.current = websocket;

    const unsubMessage = websocket.onMessage(async (message: WSMessage) => {
      console.log('[VideoInterview] Message:', message.type);

      switch (message.type) {
        case 'connected':
          setIsConnected(true);
          setError(null);

          if (message.data?.status === 'in_progress' && message.data?.current_question) {
            setStatus('in_progress');
            setCurrentQuestion(message.data.current_question);
            setQuestionNumber(message.data.question_number || 1);
            setTotalQuestions(message.data.total_questions || 5);
            setPhase('question');

            // Play audio if enabled and available
            if (voiceEnabled && message.audio_base64) {
              await playAudio(message.audio_base64);
            }
          } else if (message.data?.status === 'pending') {
            websocket.startInterview(message.data?.total_questions || 5);
          }
          break;

        case 'question':
        case 'follow_up':
          setStatus('in_progress');
          setCurrentQuestion(message.content || '');
          setQuestionNumber(message.data?.question_number || questionNumber);
          setTotalQuestions(message.data?.total_questions || totalQuestions);
          setPhase('question');
          setTranscript('');

          // Play audio if enabled
          if (voiceEnabled && message.audio_base64) {
            await playAudio(message.audio_base64);
          }
          break;

        case 'transcript':
          setTranscript(message.content || '');
          break;

        case 'complete':
          setStatus('completed');
          setPhase('waiting');
          setTimeout(() => {
            router.push(`/result/${sessionId}`);
          }, 2000);
          break;

        case 'error':
          setError(message.error || 'An error occurred');
          setPhase('question');
          break;
      }
    });

    const unsubConnect = websocket.onConnect(() => {
      console.log('[VideoInterview] Connected');
    });

    const unsubDisconnect = websocket.onDisconnect(() => {
      setIsConnected(false);
    });

    const unsubError = websocket.onError(() => {
      // Don't set error immediately
    });

    websocket.connect().catch((err) => {
      console.error('[VideoInterview] Failed to connect:', err);
      setError('Failed to connect. Please refresh the page.');
    });

    const pingInterval = setInterval(() => {
      if (websocket.isConnected) {
        websocket.ping();
      }
    }, 30000);

    return () => {
      clearInterval(pingInterval);
      unsubMessage();
      unsubConnect();
      unsubDisconnect();
      unsubError();
      websocket.disconnect();
      isInitializedRef.current = false;
    };
  }, [sessionId, router, voiceEnabled, playAudio]);

  // Start recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

        // Convert to base64
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result as string;
          const audioBase64 = base64.split(',')[1];

          // Send to server
          if (wsRef.current) {
            setPhase('processing');
            wsRef.current.sendAudioAnswer(audioBase64);
          }
        };
        reader.readAsDataURL(audioBlob);

        // Stop tracks
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setPhase('answering');
    } catch (err) {
      console.error('Failed to start recording:', err);
      alert('Could not access microphone. Please check permissions.');
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Read question aloud
  const readQuestion = () => {
    if (!wsRef.current || !currentQuestion) return;
    // Request TTS from server - for now we'll use the cached audio or request new
    // This is a simplified version - in production you might want to cache audio
    setIsSpeaking(true);
    // Simulate speaking for now since we need the audio from server
    setTimeout(() => setIsSpeaking(false), 2000);
  };

  // Skip question
  const handleSkip = () => {
    if (!wsRef.current || phase === 'processing') return;
    stopAudio();
    setPhase('processing');
    wsRef.current.skipQuestion();
  };

  // End interview
  const handleEnd = () => {
    if (!wsRef.current) return;
    if (confirm('Are you sure you want to end the interview?')) {
      stopAudio();
      wsRef.current.endInterview();
    }
  };

  const progress = totalQuestions > 0 ? (questionNumber / totalQuestions) * 100 : 0;

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="flex-shrink-0 flex items-center justify-between px-6 py-3 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-semibold">Resume Griller</h1>
          <span
            className={cn(
              'flex items-center gap-1 text-xs px-2 py-1 rounded-full',
              isConnected ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
            )}
          >
            {isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-400">
            Question {questionNumber} of {totalQuestions}
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setVoiceEnabled(!voiceEnabled)}
            className="text-gray-400 hover:text-white"
          >
            {voiceEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setCameraEnabled(!cameraEnabled)}
            className="text-gray-400 hover:text-white"
          >
            {cameraEnabled ? <Video className="w-4 h-4" /> : <VideoOff className="w-4 h-4" />}
          </Button>
          <Button variant="destructive" size="sm" onClick={handleEnd}>
            <StopCircle className="w-4 h-4 mr-1" />
            End
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col items-center justify-center p-6 gap-6">
        {/* Webcam */}
        <div className="relative w-full max-w-2xl aspect-video bg-gray-800 rounded-2xl overflow-hidden shadow-2xl">
          {cameraEnabled ? (
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="w-full h-full object-cover mirror"
              style={{ transform: 'scaleX(-1)' }}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <div className="text-center text-gray-500">
                <VideoOff className="w-16 h-16 mx-auto mb-2" />
                <p>Camera is off</p>
              </div>
            </div>
          )}

          {/* Recording indicator */}
          {isRecording && (
            <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-600 px-3 py-1 rounded-full">
              <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
              <span className="text-sm font-medium">Recording</span>
            </div>
          )}

          {/* Speaking indicator */}
          {isSpeaking && (
            <div className="absolute top-4 right-4 flex items-center gap-2 bg-blue-600 px-3 py-1 rounded-full">
              <Volume2 className="w-4 h-4 animate-pulse" />
              <span className="text-sm font-medium">Speaking</span>
            </div>
          )}
        </div>

        {/* Question Display */}
        <div className="w-full max-w-3xl">
          {status === 'completed' ? (
            <div className="text-center py-8">
              <h2 className="text-3xl font-bold text-green-400 mb-4">
                🎉 Interview Complete!
              </h2>
              <p className="text-gray-400">Redirecting to results...</p>
            </div>
          ) : phase === 'waiting' ? (
            <div className="text-center py-8">
              <Loader2 className="w-8 h-8 animate-spin mx-auto text-gray-400 mb-4" />
              <p className="text-gray-400">Preparing your interview...</p>
            </div>
          ) : phase === 'processing' ? (
            <div className="text-center py-8">
              <Loader2 className="w-8 h-8 animate-spin mx-auto text-blue-400 mb-4" />
              <p className="text-gray-400">Processing your answer...</p>
              {transcript && (
                <p className="text-sm text-gray-500 mt-2">
                  Heard: "{transcript}"
                </p>
              )}
            </div>
          ) : (
            <div className="bg-gray-800 rounded-xl p-6 shadow-lg">
              <p className="text-2xl md:text-3xl font-medium text-center leading-relaxed">
                {currentQuestion}
              </p>

              {transcript && (
                <div className="mt-4 pt-4 border-t border-gray-700">
                  <p className="text-sm text-gray-400 text-center">
                    🎤 You said: "{transcript}"
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="w-full max-w-3xl bg-red-900/50 border border-red-700 rounded-lg p-4">
            <p className="text-red-300 text-center">{error}</p>
          </div>
        )}
      </div>

      {/* Controls */}
      {status !== 'completed' && phase !== 'waiting' && (
        <div className="flex-shrink-0 bg-gray-800 border-t border-gray-700 px-6 py-4">
          <div className="max-w-3xl mx-auto">
            {/* Progress */}
            <Progress value={progress} className="h-1 mb-4" />

            {/* Buttons */}
            <div className="flex items-center justify-center gap-4">
              {!voiceEnabled && (
                <Button
                  variant="outline"
                  onClick={readQuestion}
                  disabled={isSpeaking || isRecording || phase === 'processing'}
                  className="border-gray-600 text-gray-300 hover:bg-gray-700"
                >
                  <Volume2 className="w-4 h-4 mr-2" />
                  Read Question
                </Button>
              )}

              {isRecording ? (
                <Button
                  size="lg"
                  onClick={stopRecording}
                  className="bg-red-600 hover:bg-red-700 px-8 py-6 text-lg"
                >
                  <MicOff className="w-6 h-6 mr-2" />
                  Stop Recording
                </Button>
              ) : (
                <Button
                  size="lg"
                  onClick={startRecording}
                  disabled={isSpeaking || phase === 'processing'}
                  className="bg-green-600 hover:bg-green-700 px-8 py-6 text-lg"
                >
                  <Mic className="w-6 h-6 mr-2" />
                  Start Answering
                </Button>
              )}

              <Button
                variant="outline"
                onClick={handleSkip}
                disabled={isRecording || phase === 'processing'}
                className="border-gray-600 text-gray-300 hover:bg-gray-700"
              >
                <SkipForward className="w-4 h-4 mr-2" />
                Skip
              </Button>
            </div>

            <p className="text-xs text-gray-500 mt-3 text-center">
              {isRecording
                ? 'Click "Stop Recording" when you finish your answer'
                : 'Click "Start Answering" and speak your response'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}