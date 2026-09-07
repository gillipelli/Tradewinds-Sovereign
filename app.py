"""Interactive 2026–2027 sovereign revenue event assessment."""
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='2026–2027 El Niño | Tradewinds Sovereign',page_icon='🌊',layout='wide')
st.title('2026–2027 El Niño: Pacific sovereign revenue risk')
state_path=ROOT/'artifacts/state.json'
state=json.loads(state_path.read_text()) if state_path.exists() else {}
REPORTS=Path(state['report_dir']) if state.get('risk_status')=='event_specific_conditional_assessment' else ROOT/'reports/event'
if not (REPORTS/'event_risk_metadata.json').exists():
    st.info('Run `uv run tradewinds event` to build the dated event assessment.')
    st.stop()
meta=json.loads((REPORTS/'event_risk_metadata.json').read_text())
climate=meta['climate']
st.caption(f"Assessment {climate['as_of']} · NOAA forecast {climate['forecast_issue_date']} · {len(meta['countries'])} economies · constant 2025 USD")
st.warning('Conditional event-risk study. Revenue effects are estimated historical associations, not observed losses or a validated official budget forecast. Wider economic and spending effects are outside this model.')


def read(name):
    return pd.read_csv(REPORTS/name)


def table_download(df,name):
    st.dataframe(df,hide_index=True)
    st.download_button('Download table',df.to_csv(index=False),name,'text/csv',key=name)


risk=read('event_aggregate_risk.csv')
left,right=st.columns(2)
scope=left.selectbox('Economic scope',['excluding_Australia','Pacific_islands','all_modeled'],format_func=lambda x:x.replace('_',' '))
period=right.selectbox('Assessment period',['2026–2027','2026','2027'])
row=risk[risk.scope.eq(scope)&risk.period.eq(period)].iloc[0]
a,b,c,d=st.columns(4)
a.metric('Expected revenue loss',f'${row.revenue_loss_2025usd_mean/1e6:,.0f}m')
b.metric('Probability of revenue loss',f'{row.p_revenue_loss:.1%}')
c.metric('Positive-shortfall VaR95',f'${row.var95_2025usd/1e6:,.0f}m')
d.metric('Positive-shortfall ES95',f'${row.es95_2025usd/1e6:,.0f}m')
st.caption(f"90% signed loss interval: ${row.revenue_loss_2025usd_p05/1e6:,.0f}m to ${row.revenue_loss_2025usd_p95/1e6:,.0f}m. Negative losses mean gains. Tail measures are calculated from joint draws, not summed country quantiles.")
tabs=st.tabs(['Event forecast','Sovereign revenue','Agriculture and commodities','Validation','Sensitivity and sources'])
with tabs[0]:
    p=read('event_climate.csv')
    f=go.Figure()
    f.add_trace(go.Scatter(x=p.period,y=p.p95,line={'width':0},showlegend=False))
    f.add_trace(go.Scatter(x=p.period,y=p.p05,line={'width':0},fill='tonexty',name='90% interval'))
    f.add_trace(go.Scatter(x=p.period,y=p.p50,name='Event median'))
    f.add_trace(go.Scatter(x=p.period,y=p.neutral,name='Neutral counterfactual',line={'dash':'dash'}))
    f.add_vline(x=climate['official_last_center_month'],line_dash='dot')
    f.update_layout(yaxis_title='RONI °C, centered three-month seasons',template='plotly_white')
    st.plotly_chart(f,width='stretch')
    st.write(f"Observed seasons through {climate['latest_observed_center']}; official marginal forecast through {climate['official_last_center_month']}. Later 2027 uses a statistical continuation, with separate decay and La Niña alternatives.")
    st.write(f"The no-event comparison retains history before {climate['counterfactual_start']} and sets later RONI seasons to zero. Prior La Niña and lagged effects are retained.")
    table_download(p,'event_climate.csv')
with tabs[1]:
    p=read('event_country_risk.csv')
    p=p[p.year.eq(period)].copy()
    if scope=='Pacific_islands':
        p=p[p.country.isin(['PNG','FJI','SLB','VUT','WSM','TON'])]
    elif scope=='excluding_Australia':
        p=p[p.country.ne('AUS')]
    p['mean_loss_m']=p.revenue_loss_2025usd_mean/1e6
    p['upper_m']=(p.revenue_loss_2025usd_p95-p.revenue_loss_2025usd_mean)/1e6
    p['lower_m']=(p.revenue_loss_2025usd_mean-p.revenue_loss_2025usd_p05)/1e6
    fig=px.scatter(p,x='mean_loss_m',y='country',error_x='upper_m',error_x_minus='lower_m',color='p_revenue_loss',range_color=[0,1],labels={'mean_loss_m':'Loss, millions of 2025 USD'})
    fig.add_vline(x=0,line_dash='dot')
    st.plotly_chart(fig,width='stretch')
    table_download(p,'event_country_risk.csv')
    st.write('The estimated fiscal link controls for non-agricultural growth, revenue persistence, trends and pandemic years. It is not an effective agricultural tax rate. Revenue includes grants; expenditures are not modeled.')
    with st.expander('Fiscal coefficients and IMF reference projections'):
        table_download(read('fiscal_transmission.csv'),'fiscal_transmission.csv')
        table_download(read('imf_reference.csv'),'imf_reference.csv')
        st.caption('IMF projections are nominal reference data. They are not an ENSO-free baseline; do not subtract constant-dollar losses from them.')
with tabs[2]:
    p=read('event_crop_impacts.csv')
    year=st.selectbox('Crop outcome year',[2027,2026])
    choices=st.multiselect('Commodities',sorted(p.crop.unique()),default=sorted(p.crop.unique()))
    selected=p[p.year.eq(year)&p.crop.isin(choices)]
    heat=selected.pivot(index='country',columns='crop',values='yield_effect_pct_mean')
    if not heat.empty:
        st.plotly_chart(px.imshow(heat,color_continuous_scale='RdBu',color_continuous_midpoint=0,aspect='auto',labels={'color':'Yield change %'}),width='stretch')
    st.caption('Yield effects carry forward prior-year changes. Harvested-area and local-price models are separate channels. Missing price estimates are not filled with zero. Tonnage figures are equivalents at observed production scale, not production forecasts.')
    table_download(selected,'event_crop_impacts.csv')
    table_download(read('event_crop_channel_support.csv'),'event_crop_channel_support.csv')
    st.write('Macro agricultural value added includes forestry and fishing and overlaps with crop effects. Do not sum the two.')
    table_download(risk[['period','scope','agriculture_loss_2025usd_mean','agriculture_loss_2025usd_p05','agriculture_loss_2025usd_p95']],'event_agriculture_value_added.csv')
with tabs[3]:
    st.write('Convergence does not establish predictive skill. These are conditional historical tests using realized inputs and revised data, not archived operational forecasts.')
    table_download(read('macro_validation.csv'),'macro_validation.csv')
    for kind in ['agriculture','fiscal']:
        path=REPORTS/f'{kind}_bayesian_validation.json'
        if path.exists():
            st.json(json.loads(path.read_text()))
    st.write('Check whether the climate/agriculture models actually improve RMSE and CRPS relative to their baselines. Uncertain results remain uncertain.')
    table_download(read('event_extrapolation.csv'),'event_extrapolation.csv')
    with st.expander('MCMC diagnostics'):
        st.json({k:{x:v for x,v in m.items() if x!='source_snapshot'} for k,m in meta['models'].items()})
with tabs[4]:
    p=read('event_sensitivity.csv')
    p=p[p.scope.eq(scope)&p.period.eq(period)]
    table_download(p,'event_sensitivity.csv')
    st.write('This run uses NOAA, FAOSTAT, World Bank WDI and IMF official-source data. Historical CRU weather supports the companion crop analysis. Country-average weather is not crop-weighted or live field monitoring.')
    st.write('Losses exclude event-driven prices/FX in the fiscal channel, emergency spending, and non-agricultural spillovers. Country fiscal-year alignment is approximate. Scenario alternatives are not assigned occurrence probabilities.')
    st.json(climate)
    st.download_button('Download complete portable study',(REPORTS/'index.html').read_bytes(),'2026_2027_event_study.html','text/html')
