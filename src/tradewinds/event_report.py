"""Portable event report; all conclusions derived from the accompanying run files."""
from __future__ import annotations

import html
import json
import shutil

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_event(root,out):
    meta=json.loads((out/'event_risk_metadata.json').read_text())
    climate=meta['climate']
    aggregate=pd.read_csv(out/'event_aggregate_risk.csv')
    country=pd.read_csv(out/'event_country_risk.csv')
    crops=pd.read_csv(out/'event_crop_impacts.csv')
    forecast=pd.read_csv(out/'event_climate.csv')
    validation=pd.read_csv(out/'macro_validation.csv')
    transmission=pd.read_csv(out/'fiscal_transmission.csv')
    sensitivity=pd.read_csv(out/'event_sensitivity.csv')
    extrapolation=pd.read_csv(out/'event_extrapolation.csv')
    plots=[]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=forecast.period,y=forecast.p95,line={'width':0},name='95th percentile',showlegend=False))
    fig.add_trace(go.Scatter(x=forecast.period,y=forecast.p05,line={'width':0},fill='tonexty',fillcolor='rgba(19,121,142,.2)',name='90% path interval'))
    fig.add_trace(go.Scatter(x=forecast.period,y=forecast.p50,name='Event median',line={'color':'#13798e'},customdata=forecast.source,hovertemplate='%{x}<br>RONI %{y:.2f}<br>%{customdata}<extra></extra>'))
    fig.add_trace(go.Scatter(x=forecast.period,y=forecast.neutral,name='No-event counterfactual',line={'color':'#c2752d','dash':'dash'}))
    fig.add_vline(x=climate['official_last_center_month'],line_dash='dot')
    fig.update_layout(title='The 2026–2027 climate path and its counterfactual',yaxis_title='RONI °C · centered three-month seasons',template='plotly_white',height=430,legend={'orientation':'h'})
    plots.append(fig.to_html(full_html=False,include_plotlyjs=True))
    totals=country[country.year.eq('2026–2027')].copy()
    totals['mean_m']=totals.revenue_loss_2025usd_mean/1e6
    totals['upper_m']=(totals.revenue_loss_2025usd_p95-totals.revenue_loss_2025usd_mean)/1e6
    totals['lower_m']=(totals.revenue_loss_2025usd_mean-totals.revenue_loss_2025usd_p05)/1e6
    fig=px.scatter(totals.sort_values('mean_m'),x='mean_m',y='country',error_x='upper_m',error_x_minus='lower_m',color='p_revenue_loss',color_continuous_scale='Teal',range_color=[0,1],title='Cumulative government revenue effect: uncertainty remains visible',labels={'mean_m':'Revenue loss, millions of constant 2025 USD','p_revenue_loss':'P(loss)'})
    fig.add_vline(x=0,line_dash='dot')
    fig.update_layout(template='plotly_white',height=470)
    plots.append(fig.to_html(full_html=False,include_plotlyjs=False))
    heat=crops[crops.year.eq(2027)].pivot(index='country',columns='crop',values='yield_effect_pct_mean')
    fig=px.imshow(heat,color_continuous_scale='RdBu',color_continuous_midpoint=0,aspect='auto',labels={'color':'Yield change %'},title='2027 crop yield effect, including 2026 carryover')
    fig.update_layout(template='plotly_white',height=430)
    plots.append(fig.to_html(full_html=False,include_plotlyjs=False))
    main=aggregate[aggregate.scope.eq('excluding_Australia') & aggregate.period.eq('2026–2027')].iloc[0]
    intervals=f"${main.revenue_loss_2025usd_mean/1e9:,.2f}bn (90% range ${main.revenue_loss_2025usd_p05/1e9:,.2f}bn to ${main.revenue_loss_2025usd_p95/1e9:,.2f}bn)"
    view=country[['country','year','revenue_loss_2025usd_mean','revenue_loss_2025usd_p05','revenue_loss_2025usd_p95','p_revenue_loss','revenue_loss_var95_2025usd','revenue_loss_es95_2025usd']].copy()
    for col in view:
        if 'usd' in col:
            view[col]=view[col]/1e6
    view.columns=['Country','Period','Mean loss $m','5th percentile $m','95th percentile $m','P(loss)','VaR95 $m','ES95 $m']
    agview=aggregate[['period','scope','agriculture_loss_2025usd_mean','agriculture_loss_2025usd_p05','agriculture_loss_2025usd_p95']].copy()
    agview.iloc[:,2:]=agview.iloc[:,2:]/1e9
    agview.columns=['Period','Scope','Mean VA loss $bn','5th percentile $bn','95th percentile $bn']
    sens=sensitivity[sensitivity.scope.eq('excluding_Australia') & sensitivity.period.eq('2026–2027')][['variant','revenue_loss_2025usd_mean','revenue_loss_2025usd_p05','revenue_loss_2025usd_p95','p_revenue_loss','es95_2025usd']].copy()
    sens.iloc[:,1:4]=sens.iloc[:,1:4]/1e9
    sens['es95_2025usd']/=1e9
    sens.columns=['Variant','Mean $bn','5th percentile $bn','95th percentile $bn','P(loss)','ES95 $bn']
    bayes=[]
    for kind in ['agriculture','fiscal']:
        p=out/f'{kind}_bayesian_validation.json'
        if p.exists():
            bayes.append(json.loads(p.read_text()))
    def table(df):
        return '<div class="table">'+df.to_html(index=False,float_format=lambda x:f'{x:,.3f}',border=0,escape=True)+'</div>'
    primary_limits='These are event-specific conditional risk estimates, not observed losses or a validated official budget forecast. The fiscal model estimates historical associations; it does not identify a causal tax multiplier. Neutral and event cases share non-climate shocks. Negative losses represent gains.'
    content=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026–2027 El Niño · Pacific Sovereign Revenue</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#203344;background:#f3f6f7;margin:0}}main{{max-width:1150px;margin:auto;padding:45px 25px}}h1{{font-size:38px;line-height:1.15}}h2{{margin-top:40px}}.eyebrow{{color:#13798e;font-weight:700;letter-spacing:2px}}.card{{background:white;border-radius:12px;padding:24px;margin:22px 0;box-shadow:0 2px 12px #18394b0c}}.notice{{border-left:5px solid #be823a;background:#fff8ed;padding:18px}}.table{{overflow:auto;font-size:13px}}td,th{{padding:9px;text-align:right;border-bottom:1px solid #e8edef}}th{{background:#eff5f6}}a{{color:#13798e}}.metric{{font-size:27px;font-weight:700}}.muted{{color:#556975}}.plotly-graph-div{{max-width:100%}}details{{padding:15px;background:white;margin:10px 0}}code{{background:#edf2f3;padding:3px 6px}}</style></head><body><main>
<p class="eyebrow">TRADEWINDS SOVEREIGN · EVENT ASSESSMENT</p><h1>2026–2027 El Niño and Pacific agricultural revenue risk</h1>
<p>Assessment dated {html.escape(climate['as_of'])} · NOAA outlook issued {html.escape(climate['forecast_issue_date'])} · {len(meta['countries'])} economies · {meta['simulations']:,} paired simulations.</p>
<p class="notice">{primary_limits}</p>
<div class="card"><p class="muted">2026–2027 cumulative government revenue loss · modeled economies excluding Australia</p><p class="metric">{intervals}</p><p>Probability of a revenue loss: <b>{main.p_revenue_loss:.1%}</b>. Positive-shortfall ES95: <b>${main.es95_2025usd/1e9:,.2f}bn</b>. These are constant 2025 dollars, not nominal 2027 budget dollars. Australia and Pacific-island-only aggregates remain available in the download.</p></div>
<h2>What event is being assessed?</h2><p>Observed centered seasons run through {climate['latest_observed_center']}. Official CPC marginal forecasts run through the season centered on {climate['official_last_center_month']}. The dotted boundary marks where our statistical continuation begins. Its uncertainty is not a NOAA forecast. The counterfactual preserves history before {climate['counterfactual_start']} and sets subsequent RONI seasons to zero, removing this event while retaining prior La Niña exposure.</p>{plots[0]}
<p>Overlapping seasonal forecasts are sampled with an empirical forecast-error copula. Alternative dependence, neutral decay and La Niña rebound are tested below. Annual exposure is a calendar-year approximation; it does not resolve individual planting dates.</p>
<h2>Government revenue risk</h2><p>A hierarchical fiscal model estimates the historical association between GDP-deflated revenue growth and agricultural growth contributions, controlling for non-agricultural growth, revenue persistence, country trends and pandemic years. Both scenarios start from IMF's 2025 revenue estimate. The 2027 calculation carries forward 2026 agricultural and revenue changes.</p>{plots[1]}{table(view)}
<p>All monetary values in this table are millions of constant 2025 USD. VaR95 is the 95th percentile of positive event-attributable shortfall; ES95 averages the worst 5% of those outcomes. These are neither sovereign default probabilities nor insurer premiums.</p>
<h2>Agricultural economic impact and commodity detail</h2><p>The macro outcome is agriculture, forestry and fishing value added, not gross farm sales. The crop analysis separately covers yield, harvested-area and supported local-price responses. The two routes overlap and must not be added together.</p>{table(agview)}{plots[2]}
<p><a href="event_crop_impacts.csv">Download crop effects and uncertainty</a> · <a href="event_crop_channel_support.csv">Check area/price data support</a>. Tonnage equivalents use the latest observed production scale; they are not planted-area forecasts. Missing price responses remain missing.</p>
<h2>How much confidence does the evidence support?</h2><p>Sampler convergence checks numerical estimation, not forecasting skill. The following chronological tests use revised historical data and realized climate/economic inputs. They are conditional hindcasts, not recreated forecasts available at the time. If added climate or agriculture predictors do not improve scores, the study does not claim incremental predictive skill.</p>{table(validation)}{table(pd.DataFrame(bayes))}
<p>Most extreme-event combinations have few historical analogues. The table below flags forecast exposure outside each feature's historical range; even in-range features can form unprecedented combinations.</p>{table(extrapolation)}
<details><summary>Estimated revenue transmission by country</summary>{table(transmission)}<p>These coefficients are elasticities on agriculture-share-weighted growth, not effective tax rates. Short histories and grants can make country-specific estimates weak.</p></details>
<h2>Assumptions that move the answer</h2>{table(sens)}<p>Alternatives are separate scenarios, not assigned probabilities or averaged into the central result. The zero-direct-transmission case isolates the remaining compositional channel through non-agricultural GDP weights.</p>
<h2>Comparison with institutional analyses</h2><p><a href="https://joint-research-centre.ec.europa.eu/jrc-news-and-updates/potentially-historic-el-nino-come-analysis-shows-humanitarian-toll-2026-06-15_en">JRC's June 2026 assessment</a> identifies drought exposure in Southeast Asia and Australia and differing commodity-price responses. This study adds country revenue distributions; it does not replicate JRC's physical climate model or humanitarian estimates. <a href="https://www.worldbank.org/en/topic/agriculture/publication/striking-a-balance-managing-el-nino-and-la-nina-in-the-east-asia-and-pacific-regions-agriculture">The World Bank's regional ENSO studies</a> use broader economic and policy modeling. Their GDP and welfare losses are not comparable to this narrower government-revenue channel.</p>
<p><a href="https://www.fitchratings.com/research/sovereigns/el-nino-raises-global-economic-disruption-risks-for-weaker-sovereigns-15-06-2026">Fitch's June 2026 sovereign assessment</a>: agricultural disruption can compound fiscal vulnerability; this model does not translate losses into ratings actions.</p>
<p>IMF's April 2026 country projections are supplied as reference data. They are neither an ENSO-free counterfactual nor independent validation of this project's loss estimates. No numerical agreement with an institutional 2026–2027 sovereign-revenue loss forecast is claimed.</p>
<h2>Data, limits and reproducibility</h2><p>NOAA observed RONI and official forecast quantiles; FAOSTAT production, area, producer prices and valuation; World Bank WDI sector volumes; IMF Fiscal Monitor government revenue and WEO macro projections. Historical CRU temperature, rainfall and maximum-temperature data support the complementary crop/weather analysis. Country weather averages are not crop-weighted, and annual CRU data are not live 2026 field measurements.</p>
<p>The fiscal channel omits event-driven inflation, exchange-rate changes, non-agricultural spillovers, emergency spending and tax-policy responses. Constant-price sector shares are an approximation. IMF fiscal reporting periods and country coverage differ; this annual model does not forecast monthly tax collections or reconcile each fiscal-year budget. Shared historical residual blocks preserve some cross-country risk; missing residuals require independent draws.</p>
<p>Continuous updates refresh sources and recompute the event assessment. Posterior refits occur when training inputs change. Failed, stale-forecast or nonconverged runs do not replace the last successful assessment. Provider publication delays remain real; a daily job does not create daily agricultural observations.</p>
<p><code>uv run tradewinds update</code> · <code>uv run tradewinds event --cached</code></p>
<ul><li><a href="event_country_risk.csv">Country risk results</a></li><li><a href="event_aggregate_risk.csv">Joint aggregate risk</a></li><li><a href="imf_reference.csv">Dated IMF budget-scale comparators</a></li><li><a href="event_anchors.csv">Monetary anchors and bridge years</a></li><li><a href="event_sensitivity.csv">All sensitivities</a></li><li><a href="event_risk_metadata.json">Model definitions and diagnostics</a></li><li><a href="event_climate_metadata.json">Source vintages and hashes</a></li></ul>
</main></body></html>'''
    (out/'index.html').write_text(content)
    findings=f'''# 2026–2027 event assessment\n\nAs of {climate['as_of']}, using NOAA's {climate['forecast_issue_date']} forecast.\n\nThe modeled cumulative revenue loss excluding Australia is {intervals}; P(loss)={main.p_revenue_loss:.1%}. Positive-shortfall ES95 is ${main.es95_2025usd/1e9:,.2f}bn.\n\n{primary_limits}\n\nThese estimates apply to the agriculture-linked revenue channel in constant 2025 USD. They exclude inflation/FX feedback, emergency expenditure, and broader economic spillovers. The late-2027 climate extension is model-generated.\n\nRead [the complete event study](index.html), [country results](event_country_risk.csv), and [sensitivity results](event_sensitivity.csv). Numerical convergence and chronological predictive validation are reported separately; no institutional validation or established incremental forecasting skill is asserted.\n'''
    (out/'FINDINGS.md').write_text(findings)
    # Include the exact source metadata rather than links to mutable latest data alone.
    if not (out/'source_snapshot.json').exists():
        shutil.copy2(root/'data/event/snapshot.json',out/'source_snapshot.json')
    return out/'index.html'
