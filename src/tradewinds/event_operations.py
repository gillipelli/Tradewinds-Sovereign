"""Atomic event-study refresh, posterior reuse, and validated publication."""
from __future__ import annotations

import fcntl
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .data import atomic_json, build_panel, ingest
from .event_data import ingest_event
from .event_models import (bayesian_validation, chronological_validation, fit_macro,
                           model_fingerprint)
from .event_risk import crop_event, fiscal_event, sensitivities, support_diagnostics


def reusable_macro(root, kind, cutoff=None, prior_scale=2.):
    digest=model_fingerprint(root,kind,cutoff,prior_scale)
    matches=[]
    for path in (root/'artifacts/event_models').glob('*/fit.json'):
        meta=json.loads(path.read_text())
        if meta.get('kind')==kind and meta.get('fingerprint')==digest and meta.get('diagnostics_pass'):
            matches.append(path.parent)
    return sorted(matches,key=lambda x:x.stat().st_mtime)[-1] if matches else None


def update_event(root: Path, refresh=True, refit=True):
    from .model import fit, modeling_panel
    from .operations import fingerprint
    from .event_report import render_event

    artifact=root/'artifacts'
    artifact.mkdir(exist_ok=True)
    with (artifact/'update.lock').open('w') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another update is in progress') from exc
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run=artifact/'runs'/stamp
        out=run/'reports'
        out.mkdir(parents=True)
        state_path=artifact/'state.json'
        old=json.loads(state_path.read_text()) if state_path.exists() else {}
        try:
            snapshot=ingest(root,refresh)
            build_panel(root)
            event_snapshot=ingest_event(root,refresh)
            atomic_json(run/'source_snapshot.json',{'historical':snapshot,'event':event_snapshot})
            digest=fingerprint(root,modeling_panel(root))
            crop=Path(old['model']) if old.get('model') else None
            if crop is None or old.get('training_fingerprint')!=digest:
                if not refit:
                    raise RuntimeError('Crop training changed; --no-refit cannot publish stale event results')
                crop=artifact/'models'/stamp
                fit(root,crop)
            if not json.loads((crop/'fit.json').read_text())['diagnostics_pass']:
                raise RuntimeError('Crop posterior diagnostics failed')
            models={}
            for kind in ['agriculture','fiscal']:
                folder=reusable_macro(root,kind)
                if folder is None:
                    if not refit:
                        raise RuntimeError(f'{kind} training changed; refit required')
                    folder=artifact/'event_models'/f'{kind}_{stamp}'
                    fit_macro(root,folder,kind)
                models[kind]=folder
            fiscal_event(root,models['agriculture'],models['fiscal'],out)
            crop_event(root,crop,out)
            support_diagnostics(root,models['agriculture'],out)
            chronological_validation(root,out)
            for kind in models:
                folder=reusable_macro(root,kind,2018)
                if folder is None:
                    if not refit:
                        raise RuntimeError(f'{kind} validation is stale; refit required')
                    folder=artifact/'event_models'/f'{kind}_holdout2018_{stamp}'
                    fit_macro(root,folder,kind,cutoff=2018)
                bayesian_validation(root,folder,kind,out)
            sensitivities(root,models['agriculture'],models['fiscal'],out)
            import pandas as pd
            prior_results=[]
            for scale in [.5,4.]:
                folder=reusable_macro(root,'fiscal',prior_scale=scale)
                if folder is None:
                    if not refit:
                        raise RuntimeError('Fiscal prior-sensitivity fit is stale')
                    folder=artifact/'event_models'/f'fiscal_prior{scale}_{stamp}'
                    fit_macro(root,folder,'fiscal',prior_scale=scale)
                label=f'fiscal_prior_scale_{scale}'
                result=fiscal_event(root,models['agriculture'],folder,out/'sensitivity'/label)
                result['variant']=label
                prior_results.append(result)
            pd.concat([pd.read_csv(out/'event_sensitivity.csv'),*prior_results],ignore_index=True).to_csv(out/'event_sensitivity.csv',index=False)
            # Freeze the data displayed with this successful assessment.
            (out/'data').mkdir()
            for name in ['coverage.csv','roni.csv','oni.csv','weather.csv']:
                shutil.copy2(root/'data/processed'/name,out/'data'/name)
            shutil.copy2(root/'data/event/snapshot.json',out/'source_snapshot.json')
            render_event(root,out)
            previous=Path(old['report_dir'])/'event_country_risk.csv' if old.get('report_dir') else None
            if previous is not None and previous.exists():
                import pandas as pd
                a=pd.read_csv(previous)
                b=pd.read_csv(out/'event_country_risk.csv')
                comparison=b.merge(a,on=['country','year'],suffixes=('_new','_prior'))
                comparison.to_csv(out/'assessment_revision.csv',index=False)
            state={'completed_at':stamp,'status':'complete','risk_status':'event_specific_conditional_assessment',
                   'model':str(crop),'macro_models':{k:str(v) for k,v in models.items()},
                   'report_dir':str(out),'training_fingerprint':digest,
                   'forecast_issue':event_snapshot['forecast_issue'],'target_years':[2026,2027]}
            atomic_json(run/'run.json',state)
            atomic_json(state_path,state)
            return state
        except Exception as exc:
            atomic_json(run/'run.json',{'status':'failed','error':str(exc),'prior_success_preserved':True})
            raise
