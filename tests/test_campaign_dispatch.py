"""Cross-package single-flight dispatch and bounded HTTP diagnostics."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import io
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError

import pytest

from tradewinds.agents import zai_provider
from tradewinds.agents.campaign_budget import initialize


@pytest.mark.parametrize('first_failure', [False, True])
def test_cross_package_dispatch_serializes_and_waits_before_reserving(tmp_path, monkeypatch, first_failure):
    here = Path(__file__).resolve()
    root = next((p for p in here.parents if (p / 'Tradewinds-Sovereign/src/tradewinds/agents/zai_provider.py').is_file()
                 and (p / 'Celllular-Automata-Iso/python/ashfall_agents/zai_provider.py').is_file()
                 and (p / 'Celllular-Automata-Iso/python/ashfall_agents/campaign_budget.py').is_file()), None)
    if root is None:
        pytest.skip('paired portfolio checkout unavailable')
    monkeypatch.syspath_prepend(str(root / 'Celllular-Automata-Iso/python'))
    other = importlib.import_module('ashfall_agents.zai_provider')
    ledger = initialize(tmp_path / 'budget.sqlite', 100000, {'project': 100000})
    monkeypatch.setenv('ZAI_CAMPAIGN_LEDGER', str(ledger.path))
    monkeypatch.setenv('ZAI_CAMPAIGN_PROJECT', 'project')
    monkeypatch.setenv('ZAI_CAMPAIGN_RUN', 'dispatch-test')
    monkeypatch.setattr(zai_provider, 'zai_api_key', lambda: 'test-only')
    monkeypatch.setattr(other, 'zai_api_key', lambda: 'test-only')
    first_entered = threading.Event()
    release = threading.Event()
    guard = threading.Lock()
    counts = {'active': 0, 'peak': 0, 'calls': 0}
    def request(*args, **kwargs):
        with guard:
            counts['active'] += 1
            counts['calls'] += 1
            index = counts['calls']
            counts['peak'] = max(counts['peak'], counts['active'])
        try:
            if index == 1:
                first_entered.set()
                assert release.wait(3)
                if first_failure:
                    raise TimeoutError('mock timeout')
            return io.BytesIO(json.dumps({'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
                'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ok'}}]}).encode())
        finally:
            with guard:
                counts['active'] -= 1
    monkeypatch.setattr(zai_provider, 'urlopen', request)
    monkeypatch.setattr(other, 'urlopen', request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(zai_provider.send_request, {'max_tokens': 100}, 1)
        assert first_entered.wait(3)
        second = pool.submit(other.send_request, {'max_tokens': 100}, 1)
        try:
            time.sleep(.05)
            # The waiting request has not reserved quota or entered HTTP transport.
            assert len(ledger.export_usage_report()['requests']) == 1
            assert counts['calls'] == 1
        finally:
            release.set()
        if first_failure:
            with pytest.raises(TimeoutError):
                first.result(timeout=3)
        else:
            assert first.result(timeout=3)['text'] == 'ok'
        assert second.result(timeout=3)['text'] == 'ok'
    assert counts['peak'] == 1
    assert len(ledger.export_usage_report()['requests']) == 2


def test_http_provider_details_are_bounded_and_redacted(monkeypatch):
    monkeypatch.delenv('ZAI_CAMPAIGN_LEDGER', raising=False)
    sentinel = 'test-key-must-never-leak'
    monkeypatch.setattr(zai_provider, 'zai_api_key', lambda: sentinel)
    payload = json.dumps({'error': {'code': '1302', 'message': 'rate limited ' + sentinel + 'x' * 800}}).encode()
    error = HTTPError(zai_provider.ENDPOINT, 429, 'Too Many Requests', {}, io.BytesIO(payload))
    def request(*args, **kwargs):
        raise error
    monkeypatch.setattr(zai_provider, 'urlopen', request)
    with pytest.raises(HTTPError) as caught:
        zai_provider.send_request({'max_tokens': 100}, 1)
    detail = caught.value.provider_error_detail
    assert detail['code'] == '1302'
    assert len(detail['message']) <= 512
    assert '[REDACTED]' in detail['message']
    assert sentinel not in str(caught.value) + json.dumps(detail)


def _process_request(package, ledger_path, start, active, peak, result):
    import os
    module = importlib.import_module(package + '.zai_provider')
    os.environ.update(ZAI_CAMPAIGN_LEDGER=ledger_path, ZAI_CAMPAIGN_PROJECT='project', ZAI_CAMPAIGN_RUN=package)
    module.zai_api_key = lambda: 'test-only'
    def request(*args, **kwargs):
        with active.get_lock():
            active.value += 1
            peak.value = max(peak.value, active.value)
        time.sleep(.08)
        with active.get_lock():
            active.value -= 1
        return io.BytesIO(json.dumps({'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
            'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ok'}}]}).encode())
    module.urlopen = request
    start.wait(3)
    try:
        module.send_request({'max_tokens': 100}, 1)
        result.put('ok')
    except Exception as error:
        result.put(type(error).__name__)


def test_dispatch_across_operating_system_processes(tmp_path, monkeypatch):
    import multiprocessing
    here = Path(__file__).resolve()
    root = next((p for p in here.parents if (p / 'Tradewinds-Sovereign/src/tradewinds/agents/zai_provider.py').is_file()
                 and (p / 'Celllular-Automata-Iso/python/ashfall_agents/zai_provider.py').is_file()
                 and (p / 'Celllular-Automata-Iso/python/ashfall_agents/campaign_budget.py').is_file()), None)
    if root is None:
        pytest.skip('paired portfolio checkout unavailable')
    monkeypatch.syspath_prepend(str(root / 'Tradewinds-Sovereign/src'))
    monkeypatch.syspath_prepend(str(root / 'Celllular-Automata-Iso/python'))
    ledger = initialize(tmp_path / 'process.sqlite', 100000, {'project': 100000})
    context = multiprocessing.get_context('fork')
    active, peak = context.Value('i', 0), context.Value('i', 0)
    start, result = context.Event(), context.Queue()
    children = [context.Process(target=_process_request,
        args=(package, str(ledger.path), start, active, peak, result))
        for package in ('tradewinds.agents', 'ashfall_agents')]
    for child in children:
        child.start()
    start.set()
    try:
        for child in children:
            child.join(5)
        assert all(child.exitcode == 0 for child in children)
        assert [result.get(timeout=1) for _ in children] == ['ok', 'ok']
        assert peak.value == 1
        assert ledger.export_usage_report()['charged'] == 30
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join()
