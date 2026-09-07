"""Estimated macro-agricultural response and government revenue transmission.

These are observational conditional models, not identified tax multipliers.
Revenue is GDP-deflated; grants and policy changes remain in the fiscal residual.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

from .data import annual_enso, atomic_json, config
from .event_climate import FEATURES


AG_CONTROLS = ['trend', 'lag_agri_growth', 'lag_nonagri_growth']
FISCAL_FEATURES = ['agri_contribution', 'nonagri_contribution', 'lag_revenue_growth', 'trend', 'pandemic']


def macro_panel(root):
    cfg = config(root)
    folder = root/'data/event'
    w = pd.read_csv(folder/'wdi_macro.csv').pivot(index=['country','year'], columns='indicator', values='value')
    imf = pd.read_csv(folder/'imf.csv').pivot(index=['country','year'], columns='indicator', values='value')
    p = w.join(imf[['revenue_pct_gdp']], how='left').reset_index()
    complete = pd.MultiIndex.from_product([cfg['countries'], range(1961, int(p.year.max())+1)], names=['country','year'])
    p = p.set_index(['country','year']).reindex(complete).reset_index().sort_values(['country','year'])
    # WDI component volumes in the same reference-dollar basis. Reject nonpositive residuals.
    p['nonagri_real_usd'] = p.gdp_real_usd-p.agri_real_usd
    p.loc[p.nonagri_real_usd <= 0, 'nonagri_real_usd'] = np.nan
    p['revenue_real_usd'] = p.revenue_pct_gdp/100*p.gdp_real_usd
    for prefix in ['agri','nonagri','revenue']:
        p[f'{prefix}_growth'] = p.groupby('country')[f'{prefix}_real_usd'].transform(lambda x: np.log(x.where(x>0)).diff())
        p[f'lag_{prefix}_growth'] = p.groupby('country')[f'{prefix}_growth'].shift()
    # Constant-price weights are a decomposition approximation, not nominal tax shares.
    p['agri_share'] = p.agri_real_usd/p.gdp_real_usd
    p['lag_agri_share'] = p.groupby('country').agri_share.shift()
    p['agri_contribution'] = p.lag_agri_share*p.agri_growth
    p['nonagri_contribution'] = (1-p.lag_agri_share)*p.nonagri_growth
    p['trend'] = (p.year-2000)/10
    p['pandemic'] = p.year.isin([2020,2021]).astype(float)
    enso = annual_enso(pd.read_csv(root/'data/processed/roni.csv'))
    enso = enso.sort_values('year').set_index('year')
    enso['cold_lag1'] = enso.cold.shift()
    enso['cold_lag2'] = enso.cold.shift(2)
    p = p.merge(enso[FEATURES], left_on='year', right_index=True, how='left', validate='m:1')
    p.to_csv(folder/'macro_panel.csv', index=False)
    return p


def training_data(root, kind, cutoff=None):
    cfg = config(root)
    p = macro_panel(root)
    asof = pd.Timestamp(cfg['event']['as_of']) if cfg['event'].get('as_of') else pd.Timestamp.now(tz='UTC')
    # Allow one full revision year before labels enter fitting; estimates can still anchor scenarios.
    end = cutoff if cutoff is not None else (cfg['event'].get('historical_cutoff') or asof.year-2)
    features = FEATURES+AG_CONTROLS if kind == 'agriculture' else FISCAL_FEATURES
    target = 'agri_growth' if kind == 'agriculture' else 'revenue_growth'
    p = p[(p.year <= end) & (p.year >= (1963 if kind == 'agriculture' else 1992))].dropna(subset=features+[target])
    counts = p.groupby('country').size()
    p = p[p.country.isin(counts[counts >= cfg['event']['min_macro_years']].index)].copy()
    return p, features, target


def model_fingerprint(root, kind, cutoff=None, prior_scale=2.):
    p, features, target = training_data(root, kind, cutoff)
    h = hashlib.sha256(p[['country','year',target]+features].to_csv(index=False).encode())
    h.update(Path(__file__).read_bytes())
    cfg = config(root)
    h.update(json.dumps({'sampling': {**cfg['sampling'], **cfg['event'].get('sampling', {})},
                         'prior_scale': prior_scale}, sort_keys=True).encode())
    return h.hexdigest()


def fit_macro(root, out, kind, cutoff=None, prior_scale=2.):
    cfg = config(root)
    fingerprint = model_fingerprint(root, kind, cutoff, prior_scale)
    frame, features, target = training_data(root, kind, cutoff)
    countries = sorted(frame.country.unique())
    years = sorted(frame.year.unique())
    g = frame.country.map({c:i for i,c in enumerate(countries)}).to_numpy()
    t = frame.year.map({y:i for i,y in enumerate(years)}).to_numpy()
    x = frame[features].to_numpy()
    coords = {'country':countries, 'feature':features, 'year':years, 'obs':range(len(frame))}
    out.mkdir(parents=True, exist_ok=False)
    with pm.Model(coords=coords):
        alpha = pm.Normal('alpha', 0, .05, dims='country')
        if kind == 'agriculture':
            scales = np.array([.08]*len(FEATURES)+[.02,.35,.4])
            center = pm.Normal('center', 0, scales, dims='feature')
            tau = pm.HalfNormal('tau', scales/2, dims='feature')
        else:
            # Agriculture coefficient can be negative; shrinkage does not assume a tax rate.
            scales = np.array([prior_scale,.6,.3,.02,.1])
            center = pm.Normal('center', [0,1,0,0,0], scales, dims='feature')
            tau = pm.HalfNormal('tau', scales/2, dims='feature')
        z = pm.Normal('z', 0, 1, dims=('country','feature'))
        beta = pm.Deterministic('beta', center+tau*z, dims=('country','feature'))
        sigma = pm.HalfNormal('sigma', .1 if kind=='agriculture' else .2, dims='country')
        year_sd = pm.HalfNormal('year_sd', .03 if kind=='agriculture' else .05)
        year_z = pm.Normal('year_z', 0, 1, dims='year')
        mu = alpha[g]+(beta[g]*x).sum(axis=1)+year_sd*year_z[t]
        pm.StudentT('outcome', nu=5, mu=mu, sigma=sigma[g], observed=frame[target], dims='obs')
        settings = {**cfg['sampling'], **cfg['event'].get('sampling', {})}
        idata = pm.sample(**settings, cores=min(4,settings['chains']),
                          random_seed=cfg['seed'], idata_kwargs={'log_likelihood':False})
    idata.to_netcdf(out/'posterior.nc')
    frame.to_csv(out/'training.csv', index=False)
    names = ['alpha','beta','center','tau','sigma','year_sd']
    summary = az.summary(idata, var_names=names)
    summary.to_csv(out/'parameters.csv')
    rhat = float(az.rhat(idata, var_names=names).to_array().max())
    ess = float(az.ess(idata, var_names=names).to_array().min())
    div = int(idata.sample_stats.diverging.sum())
    bfmi = float(np.min(az.bfmi(idata)))
    passed = bool(rhat<=1.01 and ess>=400 and div==0 and bfmi>=.3)
    meta = {'kind':kind, 'countries':countries, 'features':features, 'target':target,
            'training_start':int(frame.year.min()), 'training_end':int(frame.year.max()),
            'observations':len(frame), 'max_rhat':rhat, 'min_bulk_ess':ess, 'divergences':div,
            'min_bfmi':bfmi, 'diagnostics_pass':passed, 'prior_scale':prior_scale,
            'fingerprint':fingerprint,
            'source_snapshot':json.loads((root/'data/event/snapshot.json').read_text()),
            'causal_identification':False}
    atomic_json(out/'fit.json',meta)
    if not passed:
        raise RuntimeError(f'{kind} model diagnostics failed: {meta}')
    return meta


def arrays(folder):
    p = az.from_netcdf(folder/'posterior.nc').posterior.stack(sample=('chain','draw'))
    return {k:p[k].transpose('sample',...).values for k in ['alpha','beta','sigma','year_sd']}


def chronological_validation(root, out):
    """Final-vintage, conditional rolling hindcasts; never advertised as vintage forecasts."""
    rows, summary = [], []
    for kind in ['agriculture','fiscal']:
        p, features, target = training_data(root, kind)
        countries = sorted(p.country.unique())
        def design(frame, selected):
            c = np.column_stack([frame.country.eq(c).to_numpy(float) for c in countries])
            return np.column_stack([c, frame[selected].to_numpy(),
                                    np.einsum('ij,ik->ijk',c,frame[selected].to_numpy()).reshape(len(frame),-1)])
        selected = features
        baseline = AG_CONTROLS if kind=='agriculture' else [x for x in features if x!='agri_contribution']
        for year in range(2010, int(p.year.max())+1):
            train, test = p[p.year<year], p[p.year==year]
            if not len(test):
                continue
            for label, cols in [('with_climate' if kind=='agriculture' else 'with_agriculture',selected),
                                ('baseline',baseline)]:
                x, xt = design(train,cols), design(test,cols)
                # Fit scaling on training data only, including interaction columns.
                mean, sd = x.mean(0), x.std(0)
                sd[sd<1e-8]=1
                model = Ridge(alpha=25).fit((x-mean)/sd,train[target])
                prediction = model.predict((xt-mean)/sd)
                residual = train[target].to_numpy()-model.predict((x-mean)/sd)
                for i, row in enumerate(test.itertuples()):
                    errors = residual[train.country.eq(row.country)]
                    sigma = max(float(np.std(errors)),1e-4)
                    actual = getattr(row,target)
                    z = (actual-prediction[i])/sigma
                    crps = sigma*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))
                    rows.append({'kind':kind,'model':label,'year':year,'country':row.country,
                                 'actual':actual,'prediction':prediction[i], 'crps':crps,
                                 'covered90':abs(z)<=norm.ppf(.95)})
    result=pd.DataFrame(rows)
    for (kind,label), group in result.groupby(['kind','model']):
        summary.append({'kind':kind,'model':label,'n':len(group),
                        'rmse':float(np.sqrt(mean_squared_error(group.actual,group.prediction))),
                        'crps':float(group.crps.mean()),'coverage90':float(group.covered90.mean()),
                        'validation_type':'conditional final-vintage rolling hindcast; ridge challenger'})
    result.to_csv(out/'macro_rolling_predictions.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'macro_validation.csv',index=False)
    return summary


def bayesian_validation(root, folder, kind, out):
    meta=json.loads((folder/'fit.json').read_text())
    p, features, target=training_data(root,kind)
    p=p[(p.year>meta['training_end']) & p.country.isin(meta['countries'])]
    a=arrays(folder)
    g=p.country.map({c:i for i,c in enumerate(meta['countries'])}).to_numpy()
    mu=a['alpha'][:,g]+np.einsum('sij,ij->si',a['beta'][:,g],p[features].to_numpy())
    rng=np.random.default_rng(config(root)['seed'])
    # Shared year shocks within each posterior sample.
    u=rng.normal(size=(len(mu),p.year.nunique()))*a['year_sd'][:,None]
    ti=p.year.map({y:i for i,y in enumerate(sorted(p.year.unique()))}).to_numpy()
    pred=mu+u[:,ti]+rng.standard_t(5,size=mu.shape)*a['sigma'][:,g]
    actual=p[target].to_numpy()
    low,high=np.quantile(pred,[.05,.95],axis=0)
    ordered=np.sort(pred,axis=0)
    n=len(pred)
    crps=np.abs(pred-actual).mean(0)-((2*np.arange(1,n+1)-n-1)[:,None]*ordered).sum(0)/n**2
    result={'kind':kind,'training_end':meta['training_end'],'n':len(p),
            'rmse':float(np.sqrt(np.mean((pred.mean(0)-actual)**2))),
            'crps':float(crps.mean()),'coverage90':float(((actual>=low)&(actual<=high)).mean()),
            'type':'Bayesian conditional holdout with realized inputs; latest revised history'}
    pd.DataFrame({'country':p.country,'year':p.year,'actual':actual,'mean':pred.mean(0),
                  'p05':low,'p95':high,'crps':crps}).to_csv(out/f'{kind}_bayesian_holdout.csv',index=False)
    atomic_json(out/f'{kind}_bayesian_validation.json',result)
    return result
