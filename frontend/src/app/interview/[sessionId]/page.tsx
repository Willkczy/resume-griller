// frontend/src/app/interview/[sessionId]/page.tsx

import { InterviewRoom } from '@/components/interview/InterviewRoom';

interface InterviewPageProps {
  params: Promise<{
    sessionId: string;
  }>;
}

export default async function InterviewPage({ params }: InterviewPageProps) {
  const { sessionId } = await params;

  return <InterviewRoom sessionId={sessionId} />;
}