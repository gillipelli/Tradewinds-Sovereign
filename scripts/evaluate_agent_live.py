"""Run the reviewed investigation questions with a shared, explicit spend ceiling.

This collects live traces and produces an unscored human-review template. It never
equates report verification with scientific task success. No calls run by default.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

from tradewinds.assessment_bundle import freeze_assessment, safe_id
from tradewinds.data import atomic_json
from tradewinds.agents.runtime import run_agent
from tradewinds.agents.schemas import Limits
from tradewinds.agents.store import Store


def recorded_allowance(root, run_ids):
    """Count both known charges and unresolved in-flight reservations."""
    total = 0.
    for run_id in run_ids:
        path = root / 'artifacts/agent_runs' / run_id / 'state.sqlite'
        if path.exists():
            store = Store(path)
            try:
                state = store.get('checkpoints', run_id)
                if state:
                    total += state.get('cost_usd', 0.) + state.get('reserved_usd', 0.)
            finally:
                store.close()
    return total


def evaluate(root, suite, assessment_ids, evaluation_id, client, limits, *,
             split='holdout', trials=1, per_task_ceiling=.25, method='agent', task_ids=None):
    if method not in ('agent', 'fixed_evidence'):
        raise ValueError('Unknown investigation method')
    if split not in ('development', 'holdout') or type(trials) is not int or not 1 <= trials <= 3:
        raise ValueError('Choose development or holdout and one to three trials')
    if limits.spend_ceiling_usd <= 0 or not 0 < per_task_ceiling <= limits.spend_ceiling_usd:
        raise ValueError('Positive per-task and total ceilings are required')
    if len(set(assessment_ids)) != len(assessment_ids) or not assessment_ids:
        raise ValueError('Provide distinct verified assessment IDs')
    evaluation_id = safe_id(evaluation_id)
    folder = root / 'artifacts/agent_evaluations' / evaluation_id
    folder.mkdir(parents=True, exist_ok=True)
    tasks = [task for task in suite['tasks'] if task['split'] == split]
    if task_ids is not None:
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError('Provide distinct task IDs')
        unknown = set(task_ids) - {task['id'] for task in tasks}
        if unknown:
            raise ValueError(f'Task IDs outside selected split: {sorted(unknown)}')
        tasks = [task for task in tasks if task['id'] in task_ids]
    if not tasks:
        raise ValueError('No tasks in selected split')
    identifiers = [safe_id(task['id']) for task in tasks]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate task IDs')
    manifest = {'evaluation_id': evaluation_id, 'method': method, 'suite': suite, 'assessment_ids': assessment_ids,
                'provider': client.name, 'model': client.model,
                'reasoning_effort': getattr(client, 'reasoning_effort', None),
                'limits': limits.model_dump(), 'split': split, 'trials': trials,
                'per_task_ceiling': per_task_ceiling, 'task_ids': [task['id'] for task in tasks],
                'endpoint': (os.environ.get('ZAI_API_ENDPOINT', 'https://api.z.ai/api/paas/v4/chat/completions')
                             if client.name == 'zai' else getattr(client, 'endpoint', None))}
    run_ids = [safe_id(f'{evaluation_id}-{task_id}-{trial}')
               for task_id in identifiers for trial in range(1, trials + 1)]
    with (folder / 'execution.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = folder / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Evaluation settings changed; choose a new evaluation ID')
        atomic_json(manifest_path, manifest)
        rows = []
        for task in tasks:
            for trial in range(1, trials + 1):
                run_id = f'{evaluation_id}-{task["id"]}-{trial}'
                row = {'task_id': task['id'], 'trial': trial, 'run_id': run_id,
                       'required': task['required'], 'fail_if': task['fail_if'],
                       'human_success': None, 'review_notes': None}
                state = None
                remaining = max(0., limits.spend_ceiling_usd - recorded_allowance(root, run_ids))
                existing = root / 'artifacts/agent_runs' / run_id / 'state.sqlite'
                timing_path = folder / f'{run_id}.timing.json'
                if timing_path.exists():
                    row.update(json.loads(timing_path.read_text()))
                if existing.exists():
                    store = Store(existing)
                    try:
                        state = store.get('checkpoints', run_id)
                    finally:
                        store.close()
                    # A trial that was interrupted is retained as such, not rerun with
                    # a refreshed allowance. Individual resume remains an explicit action.
                    row['status'] = state['status'] if state else 'interrupted'
                    row['reason'] = state.get('reason') if state else 'No durable initial checkpoint'
                elif task.get('needs_two_assessments') and len(assessment_ids) < 2:
                    row.update(status='unavailable', reason='Requires two legitimate assessment vintages')
                elif remaining <= 0:
                    row.update(status='not_run', reason='Shared spend ceiling exhausted')
                else:
                    allowance = min(per_task_ceiling, remaining)
                    task_limits = limits.model_copy(update={'spend_ceiling_usd': allowance})
                    question = (f'Authorized assessments: {", ".join(assessment_ids)}. '
                                'Use their original dates and provided tools.\n' + task['question'])
                    started = time.monotonic()
                    previous_campaign_run = os.environ.get('ZAI_CAMPAIGN_RUN')
                    os.environ['ZAI_CAMPAIGN_RUN'] = run_id
                    try:
                        state = run_agent(root, assessment_ids, question, client,
                                          limits=task_limits, run_id=run_id,
                                          **({'method': method} if method != 'agent' else {}))
                    finally:
                        if previous_campaign_run is None:
                            os.environ.pop('ZAI_CAMPAIGN_RUN', None)
                        else:
                            os.environ['ZAI_CAMPAIGN_RUN'] = previous_campaign_run
                    row['elapsed_seconds'] = time.monotonic() - started
                    atomic_json(timing_path, {'elapsed_seconds': row['elapsed_seconds']})
                    row.update(status=state['status'], reason=state.get('reason'))
                if state:
                    for key in ('input_tokens', 'output_tokens', 'cost_usd', 'reserved_usd',
                                'turns', 'tool_count', 'repairs'):
                        row[key] = state.get(key, 0)
                row['artifact_directory'] = f'artifacts/agent_runs/{run_id}'
                rows.append(row)
                atomic_json(folder / 'results.json', {
                    'kind': 'live_trials_requiring_human_semantic_review',
                    'llm_task_success': 'not scored', 'trials': rows,
                    'charged_or_reserved_usd': recorded_allowance(root, run_ids),
                    'total_ceiling_usd': limits.spend_ceiling_usd,
                    'summary': summarize(rows)})
        review = folder / 'review_template.json'
        if not review.exists():
            atomic_json(review, {'instructions': 'Score each trial using the required/fail_if rubric; retain failures.',
                                 'trials': rows})
    return folder / 'results.json'


def summarize(rows):
    """Descriptive accounting only: completed machine checks are not human task success."""
    statuses = {}
    for row in rows:
        statuses[row['status']] = statuses.get(row['status'], 0) + 1
    return {'trial_count': len(rows), 'status_counts': statuses,
            'available_trials': sum(row['status'] != 'unavailable' for row in rows),
            'reviewed_trials': sum(row.get('human_success') is not None for row in rows),
            'input_tokens': sum(row.get('input_tokens', 0) for row in rows),
            'output_tokens': sum(row.get('output_tokens', 0) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--suite', type=Path, default=Path('evaluations/live_questions.json'))
    parser.add_argument('--assessment', action='append', required=True)
    parser.add_argument('--evaluation-id', required=True)
    parser.add_argument('--split', choices=['development', 'holdout'], default='development')
    parser.add_argument('--task-id', action='append', help='Select a task; repeat for multiple tasks')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--method', choices=['agent', 'fixed_evidence'], default='agent')
    parser.add_argument('--provider', choices=['zai', 'anthropic'], default='zai')
    parser.add_argument('--model', default='glm-5.3')
    parser.add_argument('--reasoning-effort', choices=['low', 'high', 'max'], default='high')
    parser.add_argument('--spend-ceiling', type=float, required=True)
    parser.add_argument('--per-task-ceiling', type=float, default=.25)
    parser.add_argument('--input-price', type=float, required=True)
    parser.add_argument('--output-price', type=float, required=True)
    parser.add_argument('--max-output-tokens', type=int, default=4096)
    args = parser.parse_args()
    limits = Limits(spend_ceiling_usd=args.spend_ceiling, input_usd_per_million=args.input_price,
                    output_usd_per_million=args.output_price, max_output_tokens=args.max_output_tokens)
    if args.provider == 'zai':
        from tradewinds.agents.zai_provider import ZaiClient
        client = ZaiClient(args.model, reasoning_effort=args.reasoning_effort)
    else:
        if args.model == 'glm-5.3':
            parser.error('Set an explicit Anthropic model when using that provider')
        from tradewinds.agents.provider import AnthropicClient
        client = AnthropicClient(args.model)
    root = args.root.resolve()
    for assessment_id in args.assessment:
        freeze_assessment(root, assessment_id)
    print(evaluate(root, json.loads(args.suite.read_text()), args.assessment, args.evaluation_id,
                   client, limits, split=args.split, trials=args.trials,
                   per_task_ceiling=args.per_task_ceiling, method=args.method, task_ids=args.task_id))


if __name__ == '__main__':
    main()
