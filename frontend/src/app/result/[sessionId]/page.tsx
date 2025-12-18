// frontend/src/app/result/[sessionId]/page.tsx

'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { getSessionSummary, getSession } from '@/lib/api';
import type { SessionSummary, SessionDetail } from '@/types';
import { 
  CheckCircle, 
  Clock, 
  MessageSquare, 
  HelpCircle, 
  RotateCcw,
  Home,
  Loader2
} from 'lucide-react';
import { formatDuration } from '@/lib/utils';

export default function ResultPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;
  
  const [summary, setSummary] = useState<SessionSummary | null>(null);
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [summaryData, sessionData] = await Promise.all([
          getSessionSummary(sessionId),
          getSession(sessionId),
        ]);
        setSummary(summaryData);
        setSession(sessionData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load results');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [sessionId]);

  if (loading) {
    return (
      <div className="container mx-auto max-w-2xl px-4 py-12">
        <div className="flex flex-col items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
          <p className="mt-4 text-gray-500">Loading results...</p>
        </div>
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="container mx-auto max-w-2xl px-4 py-12">
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-red-500 mb-4">{error || 'Failed to load results'}</p>
            <Link href="/">
              <Button>Go Home</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  const completionRate = (summary.questions_asked / summary.total_questions) * 100;

  return (
    <div className="container mx-auto max-w-2xl px-4 py-12">
      {/* Header */}
      <div className="text-center mb-8">
        <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
        <h1 className="text-3xl font-bold mb-2">Interview Complete!</h1>
        <p className="text-gray-600">
          Here's a summary of your mock interview session
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4 mb-8">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <MessageSquare className="w-8 h-8 text-blue-500" />
              <div>
                <p className="text-2xl font-bold">{summary.questions_asked}</p>
                <p className="text-sm text-gray-500">Questions Answered</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <HelpCircle className="w-8 h-8 text-orange-500" />
              <div>
                <p className="text-2xl font-bold">{summary.follow_ups_asked}</p>
                <p className="text-sm text-gray-500">Follow-up Questions</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <Clock className="w-8 h-8 text-purple-500" />
              <div>
                <p className="text-2xl font-bold">
                  {formatDuration(summary.duration_seconds)}
                </p>
                <p className="text-sm text-gray-500">Duration</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <CheckCircle className="w-8 h-8 text-green-500" />
              <div>
                <p className="text-2xl font-bold">{summary.answers_given}</p>
                <p className="text-sm text-gray-500">Answers Given</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Progress */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Completion Rate</CardTitle>
          <CardDescription>
            {summary.questions_asked} of {summary.total_questions} questions completed
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Progress value={completionRate} className="h-3" />
          <p className="text-right text-sm text-gray-500 mt-2">
            {completionRate.toFixed(0)}%
          </p>
        </CardContent>
      </Card>

      {/* Interview Mode */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Session Details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-500">Mode</span>
            <span className="font-medium capitalize">{summary.mode}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Status</span>
            <span className="font-medium capitalize">{summary.status}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Total Messages</span>
            <span className="font-medium">{summary.conversation_length}</span>
          </div>
        </CardContent>
      </Card>

      {/* Insights */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>💡 Insights</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {summary.follow_ups_asked === 0 ? (
            <p className="text-green-600">
              ✅ Great job! You provided detailed answers without needing follow-up questions.
            </p>
          ) : summary.follow_ups_asked <= 2 ? (
            <p className="text-yellow-600">
              ⚠️ You received {summary.follow_ups_asked} follow-up questions. Try to be more specific in your initial answers.
            </p>
          ) : (
            <p className="text-orange-600">
              🔥 You were grilled with {summary.follow_ups_asked} follow-up questions! Practice giving more detailed, specific answers with concrete examples.
            </p>
          )}
          
          <p className="text-gray-600 text-sm">
            Tip: Use the STAR method (Situation, Task, Action, Result) to structure your answers better.
          </p>
        </CardContent>
      </Card>

      {/* Actions */}
      <div className="flex gap-4">
        <Link href="/upload" className="flex-1">
          <Button variant="outline" className="w-full">
            <RotateCcw className="w-4 h-4 mr-2" />
            Try Again
          </Button>
        </Link>
        <Link href="/" className="flex-1">
          <Button className="w-full">
            <Home className="w-4 h-4 mr-2" />
            Go Home
          </Button>
        </Link>
      </div>
    </div>
  );
}