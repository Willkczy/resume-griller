// frontend/src/app/interview/[sessionId]/page.tsx

'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
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
  const mode = searchParams.get('mode') || 'video'; // 'video' or 'chat'

  useEffect(() => {
    params.then((p) => setSessionId(p.sessionId));
  }, [params]);

  if (!sessionId) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p>Loading...</p>
      </div>
    );
  }

  // Use video mode by default, fall back to chat mode
  if (mode === 'chat') {
    return <InterviewRoom sessionId={sessionId} />;
  }

  return <VideoInterviewRoom sessionId={sessionId} />;
}