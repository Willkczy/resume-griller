import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createRequire } from 'node:module';
const loadCompiledModule = createRequire(import.meta.url);

// Compile the real typed presentation/reconciliation helper; no extra test dependencies.
const output = mkdtempSync(join(tmpdir(), 'resume-griller-outcomes-'));
execFileSync(resolve('node_modules/.bin/tsc'), [
  'src/lib/interview-outcomes.ts', '--strict', '--skipLibCheck', '--target', 'ES2023',
  '--module', 'commonjs', '--moduleResolution', 'node', '--rootDir', 'src', '--outDir', output,
], { stdio: 'pipe' });
const { describeEvaluation, describeGenerationFailure, reconcileAnswerDraft,
  acceptedAnswerForQuestion, canAutomaticallyStart, isInterviewSnapshot, commandErrorMessage } = loadCompiledModule(join(output, 'lib/interview-outcomes.js'));
after(() => rmSync(output, { recursive: true, force: true }));

const snapshot = (changes = {}) => ({
  status: 'in_progress', current_question: 'Synthetic prompt?', current_question_id: 'question-1',
  parent_question_id: null, is_follow_up: false, total_questions: 3, conversation: [],
  evaluation: null, generation_failure: null, draft_answer: null,
  answer_status: 'none', allowed_actions: ['answer', 'skip', 'end'], ...changes,
});
const insufficient = { outcome: 'insufficient_evidence', reasoning: 'No specific evidence.',
  missing_elements: ['One concrete result'], suggested_follow_up: 'What happened?' };
const validZero = { outcome: 'valid', is_sufficient: false, score: 0,
  reasoning: 'A valid low evaluation.', missing_elements: ['Result'], suggested_follow_up: null };

test('valid zero is a score; insufficient evidence is separate feedback with no score', () => {
  assert.equal(describeEvaluation(validZero).score, 0);
  assert.match(describeEvaluation(validZero).title, /evaluated/);
  assert.equal(describeEvaluation(insufficient).score, null);
  assert.equal(describeEvaluation(insufficient).description, insufficient.reasoning);
  assert.deepEqual(describeEvaluation(insufficient).missingElements, insufficient.missing_elements);
});

test('every invalid/provider/unsupported outcome has distinct safe feedback and no score', () => {
  const cases = [
    ['invalid_output', 'invalid_json', 'unreadable'], ['invalid_output', 'schema_violation', 'invalid evaluation'],
    ['provider_failure', 'timeout', 'timed out'], ['provider_failure', 'rate_limited', 'request limit'],
    ['provider_failure', 'authentication', 'authenticate'], ['provider_failure', 'unavailable', 'unavailable'],
    ['provider_failure', 'unknown', 'failed'], ['unsupported_mode', 'evaluation_contract_unavailable', 'does not support'],
  ];
  for (const [outcome, code, text] of cases) {
    const result = describeEvaluation({ outcome, code });
    assert.equal(result.score, null);
    assert.match(result.description, new RegExp(text));
  }
});

test('initial reconnect restores the checkpoint failed answer', () => {
  assert.equal(reconcileAnswerDraft(snapshot({ evaluation: insufficient, draft_answer: 'Synthetic answer', answer_status: 'unscored' }), '', null, null), 'Synthetic answer');
});

test('failed answer edits, including intentional clearing, survive reconnect to stale checkpoint', () => {
  const failed = snapshot({ evaluation: insufficient, draft_answer: 'Old draft', answer_status: 'unscored' });
  assert.equal(reconcileAnswerDraft(failed, 'Edited answer with detail', 'question-1', null), 'Edited answer with detail');
  assert.equal(reconcileAnswerDraft(failed, '', 'question-1', null), '');
});

test('unconfirmed submission survives reconnect without implicit resubmission', () => {
  const pending = { questionId: 'question-1', text: 'New submitted answer' };
  assert.equal(reconcileAnswerDraft(snapshot(), pending.text, 'question-1', pending), pending.text);
  assert.equal(reconcileAnswerDraft(snapshot({ draft_answer: 'Old failed answer', answer_status: 'unscored' }), pending.text, 'question-1', pending), pending.text);
});

test('non-scored response retains submitted text; accepted same-question follow-up failure clears draft', () => {
  const pending = { questionId: 'question-1', text: 'Synthetic answer' };
  assert.equal(reconcileAnswerDraft(snapshot({ draft_answer: pending.text, answer_status: 'unscored' }), pending.text, 'question-1', pending), pending.text);
  const accepted = snapshot({ answer_status: 'accepted', allowed_actions: ['skip', 'end'], conversation: [
    { role: 'candidate', question_id: 'question-1', content: pending.text },
  ] });
  assert.equal(reconcileAnswerDraft(accepted, pending.text, 'question-1', pending), '');
  assert.equal(acceptedAnswerForQuestion(accepted.conversation, 'question-1'), pending.text);
  assert.equal(accepted.allowed_actions.includes('answer'), false);
});

test('new question clears accepted text and canonical history confirms acceptance without duplicates', () => {
  const pending = { questionId: 'question-1', text: 'Synthetic answer' };
  assert.equal(reconcileAnswerDraft(snapshot({ current_question_id: 'question-2' }), pending.text, 'question-1', pending), '');
  const acceptedHistory = snapshot({ conversation: [{ role: 'candidate', question_id: 'question-1', content: pending.text }] });
  assert.equal(reconcileAnswerDraft(acceptedHistory, pending.text, 'question-1', pending), '');
  assert.equal(acceptedHistory.conversation.length, 1);
});

test('failed startup never auto-retries; fresh pending starts at most once', () => {
  const fresh = snapshot({ status: 'pending', current_question: null, current_question_id: null, allowed_actions: ['start', 'end'] });
  assert.equal(canAutomaticallyStart(fresh, false), true);
  assert.equal(canAutomaticallyStart(fresh, true), false);
  const failure = { operation: 'question_generation', outcome: { outcome: 'provider_failure', code: 'timeout' } };
  assert.equal(canAutomaticallyStart({ ...fresh, generation_failure: failure }, false), false);
  assert.match(describeGenerationFailure(failure), /Try again/);
  assert.match(describeGenerationFailure({ ...failure, operation: 'follow_up_generation' }), /answer was accepted/);
});

test('generic command errors preserve the projected state and remain visible', () => {
  const failed = snapshot({ draft_answer: 'Old answer', answer_status: 'unscored' });
  const error = { type: 'error', error: 'Command could not be completed.', data: failed };
  assert.equal(isInterviewSnapshot(error.data), true);
  assert.equal(commandErrorMessage(error), 'Command could not be completed.');
  assert.equal(reconcileAnswerDraft(error.data, 'Edited answer', 'question-1', null), 'Edited answer');
  assert.equal(isInterviewSnapshot({ error_code: 'invalid_message' }), false);
});

test('terminal checkpoint clears draft and cannot automatically start', () => {
  for (const status of ['completed', 'cancelled']) {
    const terminal = snapshot({ status, allowed_actions: [] });
    assert.equal(reconcileAnswerDraft(terminal, 'Submitted answer', 'question-1', null), '');
    assert.equal(canAutomaticallyStart(terminal, false), false);
  }
});
