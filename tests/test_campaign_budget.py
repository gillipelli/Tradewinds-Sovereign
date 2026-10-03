from concurrent.futures import ThreadPoolExecutor
import json
from unittest.mock import patch
import pytest
from tradewinds.agents.campaign_budget import BudgetExceeded, CampaignBudget, initialize
from tradewinds.agents import zai_provider


def test_atomic_ceiling_restart(tmp_path):
    path = tmp_path / 'ledger.sqlite'
    ledger = initialize(path, 100, {'enso': 100, 'ashfall': 100})
    def reserve(index):
        try:
            return CampaignBudget(path).reserve_request(str(index), 6, 4, 'enso' if index % 2 else 'ashfall')
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(40)))
    assert sum(value is not None for value in results) == 10
    assert ledger.remaining_allowance() == 0
    assert initialize(path, 100, {'enso': 100, 'ashfall': 100}).remaining_allowance() == 0
    with pytest.raises(ValueError, match='differs'):
        initialize(path, 1000, {'enso': 100, 'ashfall': 100})


def test_settlement_uncertainty_project_limit(tmp_path):
    ledger = initialize(tmp_path / 'ledger.sqlite', 100, {'enso': 20, 'ashfall': 80})
    request = ledger.reserve_request('run', 10, 10, 'enso')
    with pytest.raises(BudgetExceeded):
        ledger.reserve_request('run', 1, 1, 'enso')
    usage = {'prompt_tokens': 3, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': 2}}
    ledger.record_usage(request, usage)
    ledger.record_usage(request, usage)
    ledger.mark_uncertain(request)
    uncertain = ledger.reserve_request('run', 5, 5, 'enso')
    ledger.mark_uncertain(uncertain)
    report = ledger.export_usage_report()
    assert report['charged'] == 15
    assert report['unresolved_reservations'] == 10
    assert report['projects']['enso']['remaining'] == 5
    assert json.loads(report['requests'][0]['usage_json']) == usage


def test_overrun_halts_campaign(tmp_path):
    ledger = initialize(tmp_path / 'ledger.sqlite', 100, {'enso': 100})
    request = ledger.reserve_request('run', 1, 1, 'enso')
    with pytest.raises(BudgetExceeded, match='exceeded'):
        ledger.record_usage(request, {'prompt_tokens': 3, 'completion_tokens': 2})
    with pytest.raises(BudgetExceeded, match='halted'):
        ledger.reserve_request('run', 1, 1, 'enso')
    assert ledger.export_usage_report()['charged'] == 5


class Reply:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return json.dumps(self.payload).encode()


@pytest.mark.parametrize('failure', [None, 'network', 'parser', 'usage'])
def test_provider_integration(tmp_path, monkeypatch, failure):
    path = tmp_path / 'ledger.sqlite'
    ledger = initialize(path, 100000, {'enso': 100000})
    monkeypatch.setenv('ZAI_CAMPAIGN_LEDGER', str(path))
    monkeypatch.setenv('ZAI_CAMPAIGN_PROJECT', 'enso')
    monkeypatch.setenv('ZAI_CAMPAIGN_RUN', 'mock-run')
    monkeypatch.setenv('ZAI_API_ENDPOINT', 'https://api.z.ai/api/coding/paas/v4/chat/completions')
    monkeypatch.setattr(zai_provider, 'zai_api_key', lambda: 'test-credential')
    payload = {'id': 'provider-id', 'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ok'}}], 'usage': {'prompt_tokens': 20, 'completion_tokens': 10}}
    if failure == 'parser':
        payload['choices'][0]['finish_reason'] = 'length'
    if failure == 'usage':
        payload['usage'] = {}
    body = {'model': 'glm-5.3', 'max_tokens': 100, 'messages': []}
    with patch.object(zai_provider, 'urlopen', return_value=Reply(payload), side_effect=TimeoutError() if failure == 'network' else None) as send:
        if failure:
            with pytest.raises((ValueError, TimeoutError)):
                zai_provider.send_request(body, 1)
        else:
            assert zai_provider.send_request(body, 1)['text'] == 'ok'
        assert send.call_args.args[0].full_url == 'https://api.z.ai/api/coding/paas/v4/chat/completions'
    report = ledger.export_usage_report()
    assert len(report['requests']) == 1
    uncertain = failure in ('network', 'usage')
    assert report['requests'][0]['status'] == ('uncertain' if uncertain else 'settled')
    assert report['charged'] == (len(json.dumps(body).encode()) + 4096 + 100 if uncertain else 30)
    assert 'test-credential' not in json.dumps(report)
    if not uncertain:
        assert json.loads(report['requests'][0]['usage_json'])['provider_response_id'] == 'provider-id'


def test_missing_campaign_metadata_prevents_http(tmp_path, monkeypatch):
    path = tmp_path / 'ledger.sqlite'
    initialize(path, 100, {'enso': 100})
    monkeypatch.setenv('ZAI_CAMPAIGN_LEDGER', str(path))
    monkeypatch.delenv('ZAI_CAMPAIGN_RUN', raising=False)
    monkeypatch.setattr(zai_provider, 'zai_api_key', lambda: 'test')
    with patch.object(zai_provider, 'urlopen') as send:
        with pytest.raises(ValueError):
            zai_provider.send_request({'max_tokens': 1}, 1)
    send.assert_not_called()
