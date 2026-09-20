// frontend/src/app/interview/[sessionId]/page.tsx

'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { getVoiceStatus } from '@/lib/api';
import { InterviewRoom } from '@/components/interview/InterviewRoom';
import { VideoInterviewRoom } from '@/components/interview/VideoInterviewRoom';

interface InterviewPageProps {
  params: Promise<{
    sessionId: string;
  }>;
}

export default function InterviewPage({ params }: InterviewPageProps) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const searchParams = useSearchParams();
  const mode = searchParams.get('mode') || 'chat';
  const [videoAvailable, setVideoAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    getVoiceStatus()
      .then((voice) => {
        if (active) {
          setVideoAvailable(voice.enabled && voice.stt_available && voice.tts_available);
        }
      })
      .catch(() => {
        if (active) setVideoAvailable(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    params.then((p) => setSessionId(p.sessionId));
  }, [params]);

  if (!sessionId || (mode === 'video' && videoAvailable === null)) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p>Loading...</p>
      </div>
    );
  }

  // Unsupported voice modes fall back to text without requesting device access.
  if (mode !== 'video' || !videoAvailable) {
    return <InterviewRoom sessionId={sessionId} />;
  }

  return <VideoInterviewRoom sessionId={sessionId} />;
}