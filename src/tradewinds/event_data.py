"""Versioned official inputs for the 2026–27 event study."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lxml import html

from .data import Store, atomic_json, config, SEASONS

CPC = 'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/outlook/'
ADVISORY = 'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml'
WEO = 'https://data.imf.org/Datasets/WEO'
QUANTILES = np.array([.05, .15, .25, .5, .75, .85, .95])
WDI = {'agri_real_usd': 'NV.AGR.TOTL.KD', 'gdp_real_usd': 'NY.GDP.MKTP.KD',
       'agri_current_usd': 'NV.AGR.TOTL.CD'}


def parse_outlook(path: Path):
    tree = html.fromstring(path.read_bytes())
    text = ' '.join(tree.itertext())
    match = re.search(r'Issued\s+([A-Za-z]+\s+20\d{2})', text)
    if not match:
        raise ValueError('CPC issue month absent')
    issue = pd.to_datetime(match[1], format='%B %Y')
    rows = []
    for tr in tree.xpath('//tr'):
        cells = [' '.join(c.itertext()).strip() for c in tr.xpath('./td|./th')]
        if not cells or not cells[0].split() or cells[0].split()[0] not in SEASONS:
            continue
        # Season cell may also contain the spelled-out months.
        try:
            values = [float(x) for x in cells[-7:]]
        except ValueError:
            continue
        if len(values) != 7 or np.any(np.diff(values) < 0):
            raise ValueError('Nonmonotone CPC forecast quantiles')
        season = cells[0].split()[0]
        month = SEASONS.index(season) + 1
        year = issue.year + int(month < issue.month)
        rows.append({'period': pd.Timestamp(year, month, 1), 'season': season,
                     **{f'q{int(q*100):02d}': v for q, v in zip(QUANTILES, values)}})
    out = pd.DataFrame(rows).sort_values('period').reset_index(drop=True)
    if len(out) != 9 or out.period.duplicated().any():
        raise ValueError(f'CPC schema changed: expected nine seasons, got {len(out)}')
    if not np.all(np.diff(out.period.dt.to_period('M').astype(int)) == 1):
        raise ValueError('Nonconsecutive forecast seasons')
    return out, issue.strftime('%Y-%m')


def ingest_event(root: Path, refresh=False):
    cfg = config(root)
    store = Store(root)
    folder = root / 'data/event'
    folder.mkdir(parents=True, exist_ok=True)
    forecast, issue = parse_outlook(store.fetch('cpc_outlook', CPC, refresh))
    advisory = store.fetch('cpc_advisory', ADVISORY, refresh)
    advisory_text = ' '.join(html.fromstring(advisory.read_bytes()).itertext())
    dates = re.findall(r'\b\d{1,2}\s+[A-Za-z]+\s+20\d{2}\b', advisory_text)
    issue_date = pd.to_datetime(dates[0], dayfirst=True) if dates else None
    if issue_date is None or issue_date.strftime('%Y-%m') != issue:
        raise ValueError('CPC advisory/outlook vintages disagree')
    imf_rows, imf_meta = [], {}
    # The public IMF endpoint accepts the standard requests client header.
    store.session.headers['User-Agent'] = requests.utils.default_user_agent()
    for label, code in {'revenue_pct_gdp': 'GGR_G01_GDP_PT', 'gdp_usd_bn': 'NGDPD',
                        'gdp_growth_pct': 'NGDP_RPCH'}.items():
        url = f'https://www.imf.org/external/datamapper/api/v2/{code}'
        body = json.loads(store.fetch(f'event_imf_{label}', url, refresh).read_text())
        if code not in body.get('values', {}):
            raise ValueError(f'Missing IMF series {code}')
        imf_meta[label] = body['indicators'][code]
        for country in cfg['countries']:
            for year, value in body['values'][code].get(country, {}).items():
                if value is not None:
                    imf_rows.append({'country': country, 'year': int(year),
                                     'indicator': label, 'value': value})
    pd.DataFrame(imf_rows).to_csv(folder / 'imf.csv', index=False)
    records = []
    for label, indicator in WDI.items():
        countries = ';'.join(cfg['countries'])
        url = f'https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}?format=json&per_page=20000&date=1961:2100'
        body = json.loads(store.fetch(f'event_{label}', url, refresh).read_text())
        if not isinstance(body, list) or len(body) != 2 or body[0].get('pages') != 1:
            raise ValueError(f'WDI schema/pagination failure: {label}')
        records.extend({'country': r['countryiso3code'], 'year': int(r['date']),
                        'indicator': label, 'value': r['value']} for r in body[1] or [] if r['value'] is not None)
    pd.DataFrame(records).to_csv(folder / 'wdi_macro.csv', index=False)
    forecast.to_csv(folder / 'cpc_quantiles.csv', index=False)
    meta = {'forecast_issue': issue, 'forecast_issue_date': str(issue_date.date()),
            'official_last_center_month': str(forecast.period.max().date()),
            'imf_metadata': imf_meta, 'sources': {k: v for k, v in store.manifest.items()
                if k.startswith('event_') or k.startswith('cpc_') or k.startswith('weo_')}}
    atomic_json(folder / 'snapshot.json', meta)
    return meta
