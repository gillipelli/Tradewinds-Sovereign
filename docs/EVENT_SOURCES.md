# Official sources and institutional comparison

| Input | Source | Role |
|---|---|---|
| Observed RONI | [NOAA CPC](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/) | Historical phases and observed 2026 event progression |
| Official RONI quantiles | [CPC outlook](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/outlook/) | Dated forecast marginals, refreshed monthly |
| Event status | [CPC advisory](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml) | Issue-date consistency and observed versus forecast status |
| Government revenue | [IMF Fiscal Monitor API v2](https://www.imf.org/external/datamapper/api/v2/GGR_G01_GDP_PT) | General government revenue, percent GDP; historical estimates and projections |
| GDP projections | [IMF WEO GDP](https://www.imf.org/external/datamapper/api/v2/NGDPD), [real growth](https://www.imf.org/external/datamapper/api/v2/NGDP_RPCH) | 2025 anchors, 2026/2027 conditional macro path, nominal comparators |
| Real agriculture VA | [WDI indicator](https://data.worldbank.org/indicator/NV.AGR.TOTL.KD) | Agriculture, forestry and fishing real volumes |
| Real GDP | [WDI indicator](https://data.worldbank.org/indicator/NY.GDP.MKTP.KD) | Real sector weights and GDP-deflated revenue construction |
| Current-price agriculture VA | [WDI indicator](https://data.worldbank.org/indicator/NV.AGR.TOTL.CD) | Exposure audit; not substituted for real growth |
| Crop production, area, producer prices, valuation | [FAOSTAT](https://www.fao.org/faostat/en/#data) | Existing verified QCL/QV/PP ingestion and quality flags |
| Historical weather | [World Bank CCKP](https://climateknowledgeportal.worldbank.org/) | CRU TS4.10 monthly country averages through 2025 |
| Global commodity/input prices | [World Bank commodity markets](https://www.worldbank.org/en/research/commodity-markets) | Existing Pink Sheet controls and complementary market analysis |

Raw provider responses are content-addressed in `data/raw`, with source URL, receipt time and SHA-256. `data/event/snapshot.json` retains exact current IMF source labels, modification dates and projection boundary. API v1 is not used: the current IMF documentation specifies v2, and v1 can return a metadata object without requested observations. Such responses are rejected.

## What external analyses can and cannot validate

[JRC, June 15, 2026](https://joint-research-centre.ec.europa.eu/jrc-news-and-updates/potentially-historic-el-nino-come-analysis-shows-humanitarian-toll-2026-06-15_en) identifies Southeast Asian/Australian drought exposure and heterogeneous food-price responses, including a changing rice-price response over the event. Its most extreme scenario extrapolates beyond historical experience. This provides directional and scope context; it is not a government-revenue loss benchmark.

[World Bank, Striking a Balance](https://www.worldbank.org/en/topic/agriculture/publication/striking-a-balance-managing-el-nino-and-la-nina-in-the-east-asia-and-pacific-regions-agriculture) models agriculture, GDP, welfare and policy interventions under ENSO in selected East Asian economies. Its broader economic outcomes cannot be equated with this project's two-year revenue channel.

[Fitch, June 15, 2026](https://www.fitchratings.com/research/sovereigns/el-nino-raises-global-economic-disruption-risks-for-weaker-sovereigns-15-06-2026) links agricultural disruption to fiscal vulnerability. No ratings impact is inferred here.

[The Fortune starting article](https://fortune.com/2026/06/17/el-nino-global-economy-trillions-in-losses/) motivates the question; official providers supply the model inputs.

No matching institutional 2026–2027 country-by-country sovereign agricultural-revenue loss distribution was found. IMF reference projections do not independently validate the event loss estimates. This project makes no claim of matching trillion-dollar global-income estimates: geography, horizon, baseline and economic outcome differ.
