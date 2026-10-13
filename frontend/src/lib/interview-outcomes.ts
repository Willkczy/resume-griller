import type {
  AnswerEvaluation, ConversationMessage, GenerationFailure,
  InterviewSnapshot, WSMessage, WSMessageData,
} from '../types';

export interface OutcomePresentation {
  title: string;
  description: string;
  score: number | null;
  missingElements: string[];
  suggestedFollowUp: string | null;
  tone: 'success' | 'warning' | 'error';
}

/** Only the valid branch can display a score, including a valid score of zero. */
export function describeEvaluation(evaluation: AnswerEvaluation): OutcomePresentation {
  const base = { score: null, missingElements: [], suggestedFollowUp: null };
  switch (evaluation.outcome) {
    case 'valid':
      return {
        title: evaluation.is_sufficient ? 'Answer evaluated' : 'Answer evaluated — more detail needed',
        description: evaluation.reasoning,
        score: evaluation.score,
        missingElements: evaluation.missing_elements,
        suggestedFollowUp: evaluation.suggested_follow_up,
        tone: evaluation.is_sufficient ? 'success' : 'warning',
      };
    case 'insufficient_evidence':
      return {
        title: 'Not enough evidence to score', description: evaluation.reasoning,
        score: null, missingElements: evaluation.missing_elements,
        suggestedFollowUp: evaluation.suggested_follow_up, tone: 'warning',
      };
    case 'invalid_output':
      return {
        ...base, title: 'Evaluation response rejected', tone: 'error',
        description: evaluation.code === 'invalid_json'
          ? 'The evaluator returned unreadable output. Your answer has not been scored.'
          : 'The evaluator returned an invalid evaluation. Your answer has not been scored.',
      };
    case 'provider_failure':
      return { ...base, title: 'Evaluation service unavailable',
        description: describeProviderFailure(evaluation.code), tone: 'error' };
    case 'unsupported_mode':
      return {
        ...base, title: 'Scoring unavailable in this mode', tone: 'warning',
        description: 'This mode does not support the evaluation contract. Your answer has not been scored. Skip this question or end the interview.',
      };
  }
}

function describeProviderFailure(code: 'timeout' | 'rate_limited' | 'authentication' | 'unavailable' | 'unknown') {
  switch (code) {
    case 'timeout': return 'The evaluation service timed out. Your answer has not been scored.';
    case 'rate_limited': return 'The evaluation service reached its request limit. Your answer has not been scored.';
    case 'authentication': return 'The evaluation service could not authenticate. Your answer has not been scored.';
    case 'unavailable': return 'The evaluation service is unavailable. Your answer has not been scored.';
    case 'unknown': return 'The evaluation service failed. Your answer has not been scored.';
  }
}

export function describeGenerationFailure(failure: GenerationFailure): string {
  const reason = failure.outcome.outcome === 'invalid_output'
    ? 'The generated question failed validation.'
    : failure.outcome.outcome === 'unsupported_mode'
      ? 'Question generation is unavailable in this mode.'
      : failure.outcome.code === 'timeout'
        ? 'The question service timed out.'
        : failure.outcome.code === 'rate_limited'
          ? 'The question service reached its request limit.'
          : failure.outcome.code === 'authentication'
            ? 'The question service could not authenticate.'
            : 'The question service is unavailable.';
  return failure.operation === 'follow_up_generation'
    ? `${reason} Your answer was accepted, but the follow-up is unavailable. Skip to continue or end the interview.`
    : `${reason} The interview has not started. Choose Try again to start explicitly, or end the interview.`;
}

export function isInterviewSnapshot(data: WSMessageData | undefined): data is WSMessageData & InterviewSnapshot {
  return !!data && typeof data.status === 'string'
    && Array.isArray(data.conversation) && Array.isArray(data.allowed_actions)
    && typeof data.answer_status === 'string';
}

export interface PendingAnswer {
  questionId: string | null;
  text: string;
}

/** Reconcile drafts with a checkpoint; never resend a command or invent accepted history. */
export function reconcileAnswerDraft(
  snapshot: InterviewSnapshot,
  localDraft: string,
  previousQuestionId: string | null,
  pending: PendingAnswer | null,
): string {
  if (snapshot.status === 'completed' || snapshot.status === 'cancelled') return '';
  if (snapshot.answer_status === 'accepted') return '';
  if (previousQuestionId !== snapshot.current_question_id) return snapshot.draft_answer ?? '';
  if (pending && snapshot.conversation.some((message) =>
    message.role === 'candidate' && message.question_id === pending.questionId
    && message.content === pending.text
  )) return '';
  if (snapshot.draft_answer !== null) {
    if (previousQuestionId === null || pending?.text === snapshot.draft_answer) return snapshot.draft_answer;
    // Reconnecting to an old failed submission must preserve newer unsent edits, even a cleared input.
    return localDraft;
  }
  // The command may never have reached the server. Keep the local draft for an explicit retry.
  return localDraft || pending?.text || '';
}

export function acceptedAnswerForQuestion(conversation: ConversationMessage[], questionId: string | null): string {
  return conversation.findLast((message) =>
    message.role === 'candidate' && message.question_id === questionId
  )?.content ?? '';
}

export function canAutomaticallyStart(snapshot: InterviewSnapshot, alreadyAttempted: boolean): boolean {
  return !alreadyAttempted && snapshot.status === 'pending'
    && !snapshot.generation_failure && snapshot.conversation.length === 0
    && snapshot.allowed_actions.includes('start');
}

export function commandErrorMessage(message: WSMessage): string | null {
  return message.type === 'error' ? message.error || message.content || 'The command failed. Your answer is retained.' : null;
}
