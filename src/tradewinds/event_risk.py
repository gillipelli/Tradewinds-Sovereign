"""Paired 2026–27 agriculture and fiscal outcomes with estimated transmission."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .data import annual_enso, atomic_json, config
from .event_climate import FEATURES, climate_paths
from .event_models import arrays, macro_panel
from .model import Design, posterior_arrays
from .risk import loss_metrics, auxiliary_responses


def summarize(values, prefix=''):
    x=np.asarray(values)
    return {f'{prefix}mean':float(x.mean()), f'{prefix}p05':float(np.quantile(x,.05)),
            f'{prefix}median':float(np.median(x)), f'{prefix}p95':float(np.quantile(x,.95))}


def residual_blocks(root, folders, countries, n, rng, dependence='shared_year'):
    """Shared consecutive historical years across both models and all countries."""
    residuals=[]
    for kind,folder in folders.items():
        meta=json.loads((folder/'fit.json').read_text())
        train=pd.read_csv(folder/'training.csv')
        a=arrays(folder)
        g=train.country.map({c:i for i,c in enumerate(meta['countries'])}).to_numpy()
        mu=a['alpha'].mean(0)[g]+np.sum(a['beta'].mean(0)[g]*train[meta['features']].to_numpy(),axis=1)
        r=train[['country','year']].copy()
        r['value']=train[meta['target']].to_numpy()-mu
        r['series']=kind+':'+r.country
        residuals.append(r)
    r=pd.concat(residuals)
    # Demean within series: innovations must not add a second fitted intercept.
    r['value']-=r.groupby('series').value.transform('mean')
    wide=r.pivot(index='year',columns='series',values='value').reindex(range(1992,int(r.year.max())+1))
    starts=rng.integers(0,len(wide)-1,n)
    result={}
    for kind in folders:
        result[kind]=np.zeros((n,2,len(countries)))
        for j,c in enumerate(countries):
            col=wide[f'{kind}:{c}'].to_numpy()
            pool=col[np.isfinite(col)]
            for t in range(2):
                x=col[starts+t].copy() if dependence=='shared_year' else rng.choice(pool,n)
                missing=~np.isfinite(x)
                x[missing]=rng.choice(pool,missing.sum())
                result[kind][:,t,j]=x
    return result


def fiscal_event(root, agriculture_model, fiscal_model, out, extension='ar2',
                 climate_dependence='empirical', residual_dependence='shared_year', onset=None,
                 fiscal_coefficient_override=None):
    cfg=config(root)
    n=cfg['risk']['simulations']
    rng=np.random.default_rng(cfg['seed']+11)
    feature,counter,paths,dates,climate_meta=climate_paths(root,out,extension,climate_dependence,onset,n)
    folders={'agriculture':agriculture_model,'fiscal':fiscal_model}
    metadata={k:json.loads((v/'fit.json').read_text()) for k,v in folders.items()}
    if not all(m['diagnostics_pass'] for m in metadata.values()):
        raise ValueError('Fiscal assessment requires passing posterior diagnostics')
    countries=sorted(set(metadata['agriculture']['countries']) & set(metadata['fiscal']['countries']))
    m=len(countries)
    posterior={}
    for kind, folder in folders.items():
        a=arrays(folder)
        s=rng.integers(0,len(a['alpha']),n)
        g=[metadata[kind]['countries'].index(c) for c in countries]
        posterior[kind]={'alpha':a['alpha'][s][:,g], 'beta':a['beta'][s][:,g]}
    if fiscal_coefficient_override is not None:
        posterior['fiscal']['beta'][:,:,0]=fiscal_coefficient_override
    innovations=residual_blocks(root,folders,countries,n,rng,residual_dependence)
    bridge_innovations=residual_blocks(root,folders,countries,n,rng,residual_dependence)
    panel=macro_panel(root)
    imf=pd.read_csv(root/'data/event/imf.csv').pivot(index=['country','year'],columns='indicator',values='value')
    hist_enso=annual_enso(pd.read_csv(root/'data/processed/roni.csv')).set_index('year')
    hist_enso['cold_lag1']=hist_enso.cold.shift()
    hist_enso['cold_lag2']=hist_enso.cold.shift(2)
    anchor_year=cfg['event']['years'][0]-1
    initial_a=np.zeros((n,m))
    initial_non=np.zeros((n,m))
    lag_a=np.zeros((n,m))
    lag_r=np.zeros((n,m))
    lag_non=np.zeros((n,m))
    initial_r=np.zeros((n,m))
    anchor_rows=[]
    for j,c in enumerate(countries):
        p=panel[panel.country.eq(c)].set_index('year')
        last=p.loc[:anchor_year].dropna(subset=['agri_real_usd','gdp_real_usd','agri_growth','nonagri_growth']).iloc[-1]
        anchor_data_year=int(last.name)
        if anchor_year-anchor_data_year>2:
            raise ValueError(f'Agriculture anchor too old: {c} {anchor_data_year}')
        alevel=np.full(n,last.agri_real_usd)
        ag=np.full(n,last.agri_growth)
        non=np.full(n,last.nonagri_growth)
        ba=posterior['agriculture']['beta'][:,j]
        for year in range(anchor_data_year+1,anchor_year+1):
            e=hist_enso.loc[year,FEATURES].to_numpy(float)
            x=np.column_stack([np.tile(e,(n,1)),np.full(n,(year-2000)/10),ag,non])
            ag=posterior['agriculture']['alpha'][:,j]+np.sum(ba*x,axis=1)+bridge_innovations['agriculture'][:,year-anchor_data_year-1,j]
            alevel*=np.exp(ag)
            non=np.full(n,np.log1p(imf.loc[(c,year),'gdp_growth_pct']/100))
        real_gdp=float(p.loc[anchor_year,'gdp_real_usd'])
        reference=imf.loc[(c,anchor_year)]
        factor=reference.gdp_usd_bn*1e9/real_gdp
        initial_a[:,j]=alevel*factor
        initial_non[:,j]=(real_gdp-alevel)*factor
        initial_r[:,j]=reference.gdp_usd_bn*1e9*reference.revenue_pct_gdp/100
        lag_a[:,j]=ag
        lag_non[:,j]=non
        previous=imf.loc[(c,anchor_year-1)]
        lag_r[:,j]=np.log(reference.revenue_pct_gdp/previous.revenue_pct_gdp)+np.log1p(reference.gdp_growth_pct/100)
        anchor_rows.append({'country':c,'agriculture_anchor_year':anchor_data_year,
                            'revenue_anchor_year':anchor_year,'revenue_anchor_status':'IMF estimate',
                            'bridge_years':anchor_year-anchor_data_year,'initial_revenue_2025usd':initial_r[0,j],
                            'imf_wdi_gdp_ratio':reference.gdp_usd_bn*1e9/real_gdp})
    if np.any(initial_non<=0):
        raise ValueError('Nonpositive non-agriculture anchor')
    states={case:{'a':initial_a.copy(),'non':initial_non.copy(),'r':initial_r.copy(),
                  'lag_a':lag_a.copy(),'lag_r':lag_r.copy(),'lag_non':lag_non.copy()}
            for case in ['event','neutral']}
    results={}
    country_rows=[]
    reference_rows=[]
    parameter_rows=[]
    b=posterior['fiscal']['beta'][:,:,0]
    for j,c in enumerate(countries):
        parameter_rows.append({'country':c,**summarize(b[:,j],'agri_revenue_elasticity_'),
                               'p_positive':float((b[:,j]>0).mean()),
                               'interpretation':'Coefficient on agriculture-share-weighted real growth; not a tax rate'})
    for t,year in enumerate(cfg['event']['years']):
        non_growth=np.array([np.log1p(imf.loc[(c,year),'gdp_growth_pct']/100) for c in countries])
        for case,e in [('event',feature[year]),('neutral',counter[year])]:
            s=states[case]
            ba=posterior['agriculture']['beta']
            controls=np.stack([np.full((n,m),(year-2000)/10),s['lag_a'],s['lag_non']],axis=2)
            ag=posterior['agriculture']['alpha']+np.einsum('nmk,nk->nm',ba[:,:,:len(FEATURES)],e)+np.sum(ba[:,:,len(FEATURES):]*controls,axis=2)+innovations['agriculture'][:,t]
            share=s['a']/(s['a']+s['non'])
            x=np.stack([share*ag,(1-share)*np.broadcast_to(non_growth,(n,m)),s['lag_r'],
                        np.full((n,m),(year-2000)/10),np.zeros((n,m))],axis=2)
            rg=posterior['fiscal']['alpha']+np.sum(posterior['fiscal']['beta']*x,axis=2)+innovations['fiscal'][:,t]
            s['a']*=np.exp(ag)
            s['non']*=np.exp(non_growth)
            s['r']*=np.exp(rg)
            s['lag_a'],s['lag_r'],s['lag_non']=ag,rg,np.broadcast_to(non_growth,(n,m))
        event,neutral=states['event'],states['neutral']
        # Freeze arrays: next year's recursion must not mutate first-year outputs.
        results[year]={k:v.copy() for k,v in {'revenue_loss':neutral['r']-event['r'],
                                            'agriculture_loss':neutral['a']-event['a'],
                                            'revenue_event':event['r'], 'revenue_neutral':neutral['r'],
                                            'agriculture_event':event['a'],'agriculture_neutral':neutral['a']}.items()}
        for j,c in enumerate(countries):
            loss=results[year]['revenue_loss'][:,j]
            shortfall=np.maximum(loss,0)
            country_rows.append({'country':c,'year':str(year),**summarize(loss,'revenue_loss_2025usd_'),
                                 **summarize(results[year]['agriculture_loss'][:,j],'agriculture_loss_2025usd_'),
                                 **summarize(event['r'][:,j],'event_revenue_2025usd_'),
                                 **summarize(neutral['r'][:,j],'neutral_revenue_2025usd_'),
                                 'p_revenue_loss':float((loss>0).mean()),
                                 'p_loss_above_1pct_revenue':float((loss>.01*neutral['r'][:,j]).mean()),
                                 'revenue_loss_var95_2025usd':float(np.quantile(shortfall,.95)),
                                 'revenue_loss_es95_2025usd':float(loss_metrics(shortfall)['es95_usd']),
                                 'revenue_loss_pct_neutral_mean':float(np.mean(loss/neutral['r'][:,j])*100)})
            reference=imf.loc[(c,year)]
            reference_rows.append({'country':c,'year':year,'imf_gdp_current_usd':reference.gdp_usd_bn*1e9,
                                   'imf_revenue_current_usd':reference.gdp_usd_bn*1e9*reference.revenue_pct_gdp/100,
                                   'imf_revenue_pct_gdp':reference.revenue_pct_gdp,
                                   'imf_real_gdp_growth_pct':reference.gdp_growth_pct,
                                   'role':'Published IMF comparator, not an ENSO-free baseline; do not subtract real losses from nominal figures'})
    # All totals preserve joint draws; never sum country quantiles.
    aggregate=[]
    for label,chosen in [(str(y),[y]) for y in cfg['event']['years']]+[('2026–2027',cfg['event']['years'])]:
        loss=sum(results[y]['revenue_loss'] for y in chosen)
        agloss=sum(results[y]['agriculture_loss'] for y in chosen)
        neutral=sum(results[y]['revenue_neutral'] for y in chosen)
        for scope,mask in [('all_modeled',np.ones(m,dtype=bool)),
                           ('Pacific_islands',np.array([c in ['PNG','FJI','SLB','VUT','WSM','TON'] for c in countries])),
                           ('excluding_Australia',np.array([c!='AUS' for c in countries]))]:
            x=loss[:,mask].sum(1)
            positive=np.maximum(x,0)
            aggregate.append({'period':label,'scope':scope,**summarize(x,'revenue_loss_2025usd_'),
                              **summarize(agloss[:,mask].sum(1),'agriculture_loss_2025usd_'),
                              'p_revenue_loss':float((x>0).mean()),
                              'p_loss_above_1pct_revenue':float((x>.01*neutral[:,mask].sum(1)).mean()),
                              'var95_2025usd':float(np.quantile(positive,.95)),
                              'es95_2025usd':float(loss_metrics(positive)['es95_usd'])})
        if len(chosen)>1:
            for j,c in enumerate(countries):
                country_rows.append({'country':c,'year':label,**summarize(loss[:,j],'revenue_loss_2025usd_'),
                                     **summarize(agloss[:,j],'agriculture_loss_2025usd_'),
                                     'p_revenue_loss':float((loss[:,j]>0).mean()),
                                     'revenue_loss_var95_2025usd':float(np.quantile(np.maximum(loss[:,j],0),.95)),
                                     'revenue_loss_es95_2025usd':float(loss_metrics(np.maximum(loss[:,j],0))['es95_usd']),
                                     'revenue_loss_pct_neutral_mean':float(np.mean(loss[:,j]/neutral[:,j])*100)})
    pd.DataFrame(country_rows).to_csv(out/'event_country_risk.csv',index=False)
    pd.DataFrame(aggregate).to_csv(out/'event_aggregate_risk.csv',index=False)
    pd.DataFrame(reference_rows).to_csv(out/'imf_reference.csv',index=False)
    pd.DataFrame(anchor_rows).to_csv(out/'event_anchors.csv',index=False)
    pd.DataFrame(parameter_rows).to_csv(out/'fiscal_transmission.csv',index=False)
    np.savez_compressed(out/'event_draws.npz',countries=np.array(countries),
                        **{f'{key}_{year}':val for year,vals in results.items() for key,val in vals.items()})
    atomic_json(out/'event_risk_metadata.json',{
        'climate':climate_meta,'models':metadata,'countries':countries,
        'omitted_countries':sorted(set(cfg['countries'])-set(countries)),
        'simulations':n,'units':'constant 2025 USD, GDP-deflated; fixed 2025 exchange rate',
        'scope':'Agriculture, forestry and fishing value added; conditional government revenue transmission',
        'revenue_includes_grants':True,'residual_dependence':residual_dependence,
        'nonagriculture_scenario':'IMF total real GDP growth used as an external non-agriculture growth proxy, shared across cases',
        'price_and_fx_response_modeled':False,'causal_identification':False,
        'historical_validation_is_conditional':True,
        'risk_definition':'Positive = neutral revenue minus event revenue. Gains retained in signed estimates; VaR/ES apply to positive shortfall.',
        'not_total_economy_loss':True,'not_sovereign_default_probability':True,
        'fiscal_coefficient_override':fiscal_coefficient_override})
    return pd.DataFrame(aggregate)


def crop_event(root, crop_model, out):
    """Dated yield effects; fixed-area production equivalents clearly distinguished."""
    cfg=config(root)
    n=cfg['risk']['simulations']
    event,neutral,_,_,_=climate_paths(root,out,n=n)
    d=Design.load(crop_model/'design.json')
    a=posterior_arrays(crop_model)
    rng=np.random.default_rng(cfg['seed']+22)
    draw=rng.integers(0,len(a['alpha']),n)
    beta=a['beta'][draw]
    gamma=a['gamma'][draw]
    # Derivative of standardized lag-yield control, not the unscaled coefficient.
    ar=gamma[:,d.names.index('lag_yield_growth')]/d.std['lag_yield_growth']
    previous=np.zeros((n,len(d.series)))
    level=np.zeros_like(previous)
    training=pd.read_csv(crop_model/'training.csv')
    aux,_,support=auxiliary_responses(root,training,d.series,200,rng)
    support.to_csv(out/'event_crop_channel_support.csv',index=False)
    aux_draw=rng.integers(0,200,n)
    area_level=np.zeros_like(previous)
    price_level=np.zeros_like(previous)
    p=pd.read_csv(root/'data/processed/panel.csv')
    rows=[]
    for year in cfg['event']['years']:
        # Crop model has five ENSO terms; macro model additionally has cold lags.
        delta=event[year][:,:5]-neutral[year][:,:5]
        growth=np.einsum('nsk,nk->ns',beta,delta)+ar[:,None]*previous
        level+=growth
        area_level+=np.einsum('nsk,nk->ns',aux['area_growth'][aux_draw],delta)
        price_level+=np.einsum('nsk,nk->ns',aux['price_growth'][aux_draw],delta)
        ratio=np.expm1(level)
        for j,series in enumerate(d.series):
            anchor=p[p.series.eq(series)].dropna(subset=['production_t','area_ha']).sort_values('year').iloc[-1]
            country,crop=series.split(':')
            price_supported=bool(support.loc[support.series.eq(series) & support.outcome.eq('price_growth'),'estimated'].iloc[0])
            area_supported=bool(support.loc[support.series.eq(series) & support.outcome.eq('area_growth'),'estimated'].iloc[0])
            rows.append({'country':country,'crop':crop,'year':year,
                         **summarize(ratio[:,j]*100,'yield_effect_pct_'),
                         'p_yield_loss':float((ratio[:,j]<0).mean()),'anchor_year':int(anchor.year),
                         'anchor_production_t':anchor.production_t,
                         'area_response_estimated':area_supported,'price_response_estimated':price_supported,
                         **(summarize(np.expm1(level[:,j]+area_level[:,j])*100,'production_effect_pct_') if area_supported else {}),
                         **(summarize(np.expm1(price_level[:,j])*100,'local_usd_price_effect_pct_') if price_supported else {}),
                         **(summarize(np.expm1(level[:,j]+area_level[:,j]+price_level[:,j])*100,'gross_receipts_effect_pct_') if price_supported and area_supported else {}),
                         **summarize(-ratio[:,j]*anchor.production_t,'fixed_area_anchor_output_loss_t_'),
                         'tonnage_interpretation':'Equivalent at latest observed production; not a forecast of planted area or 2026/27 output',
                         'scope':'Crop yield, not fiscal revenue; do not add to macro value-added losses'})
        previous=growth
    pd.DataFrame(rows).to_csv(out/'event_crop_impacts.csv',index=False)
    return rows


def support_diagnostics(root, agriculture_model, out):
    """Flag marginal extrapolation; correlated combinations may also be unsupported."""
    train=pd.read_csv(agriculture_model/'training.csv').drop_duplicates('year')
    events,neutral,_,_,_=climate_paths(root,out)
    rows=[]
    for year, e in events.items():
        for i, feature in enumerate(FEATURES):
            lo,hi=train[feature].min(),train[feature].max()
            rows.append({'year':year,'feature':feature,'training_min':lo,'training_max':hi,
                         'forecast_mean':e[:,i].mean(),
                         'probability_outside_training_range':float(((e[:,i]<lo)|(e[:,i]>hi)).mean()),
                         'neutral_mean':neutral[year][:,i].mean()})
    pd.DataFrame(rows).to_csv(out/'event_extrapolation.csv',index=False)


def sensitivities(root, agriculture_model, fiscal_model, out):
    """Scenario alternatives are not probability-weighted into the central estimate."""
    variants={'neutral_late_2027':{'extension':'neutral_decay'},
              'la_nina_late_2027':{'extension':'la_nina_rebound'},
              'independent_forecast_errors':{'climate_dependence':'independent'},
              'aligned_forecast_errors':{'climate_dependence':'comonotonic'},
              'independent_economic_shocks':{'residual_dependence':'independent'},
              'onset_june':{'onset':'2026-06-01'},
              'zero_direct_fiscal_transmission':{'fiscal_coefficient_override':0.}}
    tables=[]
    for label,options in variants.items():
        folder=out/'sensitivity'/label
        result=fiscal_event(root,agriculture_model,fiscal_model,folder,**options)
        result['variant']=label
        tables.append(result)
    main=pd.read_csv(out/'event_aggregate_risk.csv')
    main['variant']='central'
    tables.append(main)
    result=pd.concat(tables,ignore_index=True)
    result.to_csv(out/'event_sensitivity.csv',index=False)
    return result
