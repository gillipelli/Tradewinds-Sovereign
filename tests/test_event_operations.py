"""The event pipeline must never replace a successful assessment on a failed refresh."""
import json

import pytest

from tradewinds import event_operations


def test_event_failure_preserves_prior_published_state(tmp_path,monkeypatch):
    artifact=tmp_path/'artifacts'
    artifact.mkdir()
    prior={'status':'complete','report_dir':'prior_report','risk_status':'event_specific_conditional_assessment'}
    path=artifact/'state.json'
    path.write_text(json.dumps(prior))
    def fail(*args):
        raise ValueError('Official source schema changed')
    monkeypatch.setattr(event_operations,'ingest',fail)
    with pytest.raises(ValueError,match='schema'):
        event_operations.update_event(tmp_path)
    assert json.loads(path.read_text())==prior
    statuses=[json.loads(p.read_text()) for p in (artifact/'runs').glob('*/run.json')]
    assert statuses==[{'status':'failed','error':'Official source schema changed','prior_success_preserved':True}]
