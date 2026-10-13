import type { AnswerEvaluation, GenerationFailure } from '@/types';
import { describeEvaluation, describeGenerationFailure } from '@/lib/interview-outcomes';
import { cn } from '@/lib/utils';

interface EvaluationNoticeProps {
  evaluation: AnswerEvaluation | null;
  generationFailure: GenerationFailure | null;
  canAnswer: boolean;
  dark?: boolean;
  evaluatedQuestion?: string | null;
  isPreviousAnswer?: boolean;
}

export function EvaluationNotice({ evaluation, generationFailure, canAnswer, dark = false, evaluatedQuestion, isPreviousAnswer = false }: EvaluationNoticeProps) {
  const outcome = evaluation ? describeEvaluation(evaluation) : null;
  if (!outcome && !generationFailure) return null;
  return (
    <div className="space-y-3" aria-live="polite">
      {outcome && (
        <div className={cn('rounded-lg border p-4 text-sm', dark
          ? 'border-gray-600 bg-gray-800 text-gray-100'
          : outcome.tone === 'error' ? 'border-red-200 bg-red-50 text-red-900'
            : outcome.tone === 'warning' ? 'border-amber-200 bg-amber-50 text-amber-900'
              : 'border-green-200 bg-green-50 text-green-900')}>
          {(evaluatedQuestion || isPreviousAnswer) && <p className="mb-2 text-xs">{evaluatedQuestion ? `Feedback for: ${evaluatedQuestion}` : 'Feedback for the previous answer'}</p>}
          <p className="font-semibold">{outcome.title}</p>
          {outcome.score !== null && <p className="mt-1 font-semibold">Score: {Math.round(outcome.score * 100)} / 100</p>}
          <p className="mt-1 whitespace-pre-wrap">{outcome.description}</p>
          {outcome.missingElements.length > 0 && (
            <div className="mt-2">
              <p className="font-medium">Missing evidence</p>
              <ul className="list-disc pl-5">{outcome.missingElements.map((element, index) => <li key={index}>{element}</li>)}</ul>
            </div>
          )}
          {outcome.suggestedFollowUp && <p className="mt-2">Consider: {outcome.suggestedFollowUp}</p>}
          {outcome.score === null && canAnswer && <p className="mt-2">Your answer is retained below. Edit and submit it explicitly, skip this question, or end the interview.</p>}
        </div>
      )}
      {generationFailure && (
        <div className={cn('rounded-lg border p-4 text-sm', dark ? 'border-amber-700 bg-amber-900/30 text-amber-200' : 'border-amber-200 bg-amber-50 text-amber-900')}>
          <p className="font-semibold">{generationFailure.operation === 'follow_up_generation' ? 'Follow-up unavailable' : 'Interview could not start'}</p>
          <p className="mt-1">{describeGenerationFailure(generationFailure)}</p>
        </div>
      )}
    </div>
  );
}
