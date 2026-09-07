"""Event-specific temporal and actuarial contracts, independent of live providers."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tradewinds.event_climate import annual_features, error_copula, quantile_transform
from tradewinds.event_data import QUANTILES, parse_outlook


def test_cpc_quantiles_and_year_rollover(tmp_path):
    seasons=['JAS','ASO','SON','OND','NDJ','DJF','JFM','FMA','MAM']
    rows=''.join('<tr><td>'+s+'</td>'+''.join(f'<td>{x}</td>' for x in range(7))+'</tr>' for s in seasons)
    path=tmp_path/'cpc.html'
    path.write_text('<h2>Issued August 2026</h2><table><tr><td></td></tr>'+rows+'</table>')
    frame,issue=parse_outlook(path)
    assert issue=='2026-08'
    assert frame.period.iloc[0]==pd.Timestamp('2026-08-01')
    assert frame.period.iloc[-1]==pd.Timestamp('2027-04-01')
    np.testing.assert_allclose(quantile_transform(norm.ppf(QUANTILES),np.arange(7)),np.arange(7))
    assert quantile_transform(norm.ppf(.999),np.arange(7))>6


def test_cpc_rejects_changed_or_nonmonotone_source(tmp_path):
    path=tmp_path/'bad.html'
    path.write_text('<h2>Issued August 2026</h2><table><tr><td>JAS</td>'+''.join(f'<td>{x}</td>' for x in [0,1,2,1,4,5,6])+'</tr></table>')
    with pytest.raises(ValueError,match='Nonmonotone'):
        parse_outlook(path)


def test_2027_carryover_preserves_pre_event_la_nina():
    dates=pd.date_range('2024-01-01','2027-12-01',freq='MS')
    paths=np.zeros((3,len(dates)))
    paths[:,dates.year==2025]=-1
    paths[:,dates.year==2026]=2
    e2026=annual_features(paths,dates,2026)
    e2027=annual_features(paths,dates,2027)
    np.testing.assert_equal(e2026[:,0],2)
    np.testing.assert_equal(e2026[:,5],1)
    np.testing.assert_equal(e2027[:,1],2)
    np.testing.assert_equal(e2027[:,6],1)
    with pytest.raises(ValueError,match='Incomplete'):
        annual_features(paths[:,:-1],dates[:-1],2027)


def test_copula_is_positive_definite_and_temporally_dependent():
    rng=np.random.default_rng(51)
    y=np.zeros(800)
    for i in range(2,len(y)):
        y[i]=1.4*y[i-1]-.5*y[i-2]+rng.normal(0,.1)
    corr,n=error_copula(y)
    assert n>=30
    assert np.linalg.eigvalsh(corr).min()>0
    assert corr[0,1]>.5
    np.testing.assert_allclose(corr.diagonal(),1)


def test_assessment_rejects_future_and_stale_forecasts(tmp_path, monkeypatch):
    from tradewinds import event_climate
    (tmp_path/'data/processed').mkdir(parents=True)
    (tmp_path/'data/event').mkdir()
    pd.DataFrame({'period':['2026-07-01'],'assumed_available_at':['2026-09-05'],'enso':[1.4]}).to_csv(tmp_path/'data/processed/roni.csv',index=False)
    pd.DataFrame({'period':['2026-08-01']}).to_csv(tmp_path/'data/event/cpc_quantiles.csv',index=False)
    cfg={'seed':1,'risk':{'simulations':2},'event':{'counterfactual_start':'2026-05-01','as_of':'2026-09-06','max_forecast_age_days':45}}
    monkeypatch.setattr(event_climate,'config',lambda _:cfg)
    path=tmp_path/'data/event/snapshot.json'
    path.write_text(json.dumps({'forecast_issue_date':'2026-09-10'}))
    with pytest.raises(ValueError,match='after assessment'):
        event_climate.climate_paths(tmp_path,tmp_path/'out')
    path.write_text(json.dumps({'forecast_issue_date':'2026-06-11'}))
    with pytest.raises(ValueError,match='stale'):
        event_climate.climate_paths(tmp_path,tmp_path/'out')
