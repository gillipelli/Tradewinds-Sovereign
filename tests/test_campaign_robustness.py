"""Offline fault injection and shared-ledger interoperability gates."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

import pytest

from tradewinds.agents import credentials, zai_provider
from tradewinds.agents.campaign_budget import BudgetExceeded, CampaignBudget, initialize


@pytest.mark.parametrize('code', [401, 429, 500])
def test_http_failure_preserves_reservation_without_credentials(tmp_path, monkeypatch, capsys, code):
    path = tmp_path / 'campaign.sqlite'
    ledger = initialize(path, 100000, {'project': 100000})
    monkeypatch.setenv('ZAI_CAMPAIGN_LEDGER', str(path))
    monkeypatch.setenv('ZAI_CAMPAIGN_PROJECT', 'project')
    monkeypatch.setenv('ZAI_CAMPAIGN_RUN', 'http-fault')
    sentinel = 'test-only-secret-never-store'
    monkeypatch.setattr(zai_provider, 'zai_api_key', lambda: sentinel)
    error = HTTPError(zai_provider.ENDPOINT, code, 'mock API failure', {}, io.BytesIO(b'{"error":"mock"}'))
    with patch.object(zai_provider, 'urlopen', side_effect=error) as request:
        with pytest.raises(HTTPError) as caught:
            zai_provider.send_request({'max_tokens': 100, 'messages': []}, 1)
    request.assert_called_once()
    report = ledger.export_usage_report()
    assert report['requests'][0]['status'] == 'uncertain'
    assert report['charged'] == report['requests'][0]['reserved']
    output = capsys.readouterr()
    assert sentinel not in json.dumps(report) + str(caught.value) + output.out + output.err
    assert sentinel.encode() not in path.read_bytes()


@pytest.mark.parametrize('environment, expected', [('environment-key', 'environment-key'), ('', ''), (None, 'file-key')])
def test_credential_precedence_and_no_shell_execution(tmp_path, monkeypatch, environment, expected):
    env_file = tmp_path / '.env'
    env_file.write_text('OTHER=$(touch dangerous)\nZAI_API_KEY="file-key"\n')
    monkeypatch.setattr(credentials, 'ENV_FILE', env_file)
    if environment is None:
        monkeypatch.delenv('ZAI_API_KEY', raising=False)
    else:
        monkeypatch.setenv('ZAI_API_KEY', environment)
    with patch('subprocess.run') as run:
        assert credentials.zai_api_key() == expected
    run.assert_not_called()
    assert not (tmp_path / 'dangerous').exists()


def test_cross_project_sqlite_interoperability(tmp_path):
    # Optional integration gate when both portfolio repositories are checked out.
    here = Path(__file__).resolve()
    root = next((p for p in here.parents if (p / 'Tradewinds-Sovereign/src/tradewinds/agents/zai_provider.py').is_file()
                 and (p / 'Celllular-Automata-Iso/python/ashfall_agents/zai_provider.py').is_file()
                 and (p / 'Celllular-Automata-Iso/python/ashfall_agents/campaign_budget.py').is_file()), None)
    if root is None:
        pytest.skip('paired portfolio checkout unavailable')
    other = root / 'Celllular-Automata-Iso/python/ashfall_agents/campaign_budget.py'
    spec = importlib.util.spec_from_file_location('other_campaign_ledger', other)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / 'shared.sqlite'
    ledger = initialize(path, 120, {'enso': 100, 'ashfall': 100})
    def reserve(index):
        package = CampaignBudget if index % 2 else module.CampaignBudget
        try:
            return package(path).reserve_request(str(index), 5, 5, 'enso' if index % 2 else 'ashfall')
        except (BudgetExceeded, module.BudgetExceeded):
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(40)))
    assert len([r for r in results if r]) == 12
    assert module.CampaignBudget(path).export_usage_report() == ledger.export_usage_report()
    assert ledger.remaining_allowance() == 0


def test_truncated_response_reconciles_actual_billed_usage(tmp_path, monkeypatch):
    path = tmp_path / 'campaign.sqlite'
    ledger = initialize(path, 100000, {'project': 100000})
    for key, value in {'ZAI_CAMPAIGN_LEDGER': str(path), 'ZAI_CAMPAIGN_PROJECT': 'project', 'ZAI_CAMPAIGN_RUN': 'truncated'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(zai_provider, 'zai_api_key', lambda: 'test-secret')
    payload = {'id': 'truncated-response', 'usage': {'prompt_tokens': 12, 'completion_tokens': 80},
               'choices': [{'finish_reason': 'length', 'message': {'role': 'assistant', 'content': 'partial'}}]}
    response = io.BytesIO(json.dumps(payload).encode())
    with patch.object(zai_provider, 'urlopen', return_value=response):
        with pytest.raises(ValueError, match='incomplete'):
            zai_provider.send_request({'max_tokens': 100}, 1)
    report = ledger.export_usage_report()
    assert report['charged'] == 92
    assert report['unresolved_reservations'] == 0
    assert report['requests'][0]['status'] == 'settled'
    assert json.loads(report['requests'][0]['usage_json'])['provider_response_id'] == 'truncated-response'
