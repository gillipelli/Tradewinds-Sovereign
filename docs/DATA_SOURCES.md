# Data sources and selection rationale

All implemented economic and climate observations come directly from official providers.
The Fortune article supplied the topic, not training labels or a numeric loss target.
Exact retrieval timestamps, URLs, byte counts, last-modified headers, and SHA-256 hashes are
stored in `data/raw/manifest.json`. Raw files are immutable by content hash.

| Source | Implemented variables | Purpose and quality limits |
|---|---|---|
| [NOAA CPC RONI](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/) | Three-month relative Niño 3.4 index, 1950 onward | Primary ENSO exposure; latest estimates and historical values are revisable. Full current page is archived. |
| [NOAA CPC ONI](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt) | Historical ONI | Separate sensitivity series; never spliced with RONI. |
| [FAOSTAT QCL](https://www.fao.org/faostat/en/#data/QCL) | Production tonnes, harvested hectares, source flags | Country–primary-crop panel; yield derived as t/ha to avoid yield-unit changes. |
| [FAOSTAT QV](https://www.fao.org/faostat/en/#data/QV) | Gross production value in current thousand USD | Exact crop exposure values; converted ×1,000 to USD. These are gross values, not value added or taxes. |
| [FAOSTAT PP](https://www.fao.org/faostat/en/#data/PP) | Annual and monthly farmgate prices; USD/tonne used for risk response | Exact primary-crop correspondence. Annual rows are explicitly selected; monthly and annual observations never pooled. Sparse series get a disclosed fixed-price sensitivity. |
| [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets) | Monthly global commodity prices; crude oil and urea controls | Global trade benchmarks differ from farmgate prices and sometimes from the primary crop product. Analyzed separately. |
| [World Bank WDI](https://data.worldbank.org/) | GDP current USD, real GDP growth, revenue excluding grants/GDP, agriculture value added/GDP | Fiscal context and lagged activity controls. Matched-year GDP and revenue required; source government coverage differs by country. Agriculture aggregate includes forestry and fishing. |
| [CRU TS via World Bank CCKP](https://climateknowledgeportal.worldbank.org/download-data) | Monthly `tas`, `pr`, `tasmax`, CRU TS 4.10, 1901–2025 | Station-based gridded historical climate aggregated over national land; national averages can miss crop-growing conditions. |

FAOSTAT bulk URLs are discovered from its [official catalog](https://bulks-faostat.fao.org/production/datasets_E.json),
not copied from unofficial mirrors. The parser preserves provider item names, units, and flags.

## Weather interpretation

`tas`: mean temperature (°C); `pr`: monthly rainfall accumulation (mm); `tasmax`: monthly mean
of daily maximum temperature (°C). Annual rainfall sums twelve monthly values. Driest-month
rain and hottest-month mean maximum temperature are **proxies**, not SPI/SPEI, daily heatwaves,
or flood counts. Incomplete variable-years are not treated as complete annual observations.
Country centering and scaling in weather models are fitted within training data only.

The [CCKP API documentation](https://climateknowledgeportal.worldbank.org/download-data) defines
country aggregation endpoints. The [CRU producer](https://crudata.uea.ac.uk/cru/data/hrg/) documents
the underlying historical station-based dataset. A pinned full-history version prevents mixing
release methodologies. Inspect coverage before changing `weather.version` and `weather.period`.

## Evidence grades and exclusions

Eligibility requires ≥30 observed annual growth values and crop production no more than three
years behind the newest crop year. FAO flag `A` is tracked as official. Other flags remain in the
normalized data and are not reclassified as independent official measurements. Grades are
mostly official (≥80%), mixed (50–80%), or mostly estimated/imputed (<50%). These grades concern
production records, not an endorsement of every yield or price observation.

Risk exposure requires a positive exact-crop valuation in the common latest valuation year.
Missing valuations are excluded and exported. Old Papua New Guinea values and missing Solomon
Islands values must not be silently carried into a current USD portfolio. Fiscal buffer calculations
require GDP and revenue ratios from the same year, no more than two years before the valuation year.
Missing fiscal denominators stay missing; absolute sensitivity values do not imply a known budget ratio.

## Refresh frequencies and availability

NOAA indices and commodity prices are generally monthly; crop output, fiscal indicators, and
CRU historical releases are annual with publication lags. The pipeline checks sources daily
when scheduled, using ETags and content hashes. `retrieved_at` is the project's knowledge time,
not a reconstructed historical provider publication date. Centered ENSO seasons have a conservative
assumed availability date after the final month; it is labeled as assumed, not provider-certified.

## Sources evaluated but not fabricated into this release

- [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means):
  useful for recent weather and eventual crop-area aggregation; a dedicated gridded extraction
  pipeline and spatial weights are not implemented here.
- [CHIRPS](https://www.chc.ucsb.edu/data/chirps): suitable rainfall cross-check. Its producer
  is transitioning from version 2 to version 3; a reviewed full-history version is needed.
- Country finance ministries, agricultural agencies, IMF GFS, and UN Comtrade: priorities for
  calibrated fiscal capture and trade channels. No invented national tax rates are substituted.
- [Callahan & Mankin (2023)](https://doi.org/10.1126/science.adf2983): motivates investigating persistent
  ENSO economic associations. Its global macroeconomic estimates are not crop losses or fiscal targets.

## Terms

Keep provider attribution with exports. Consult [World Bank dataset terms](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets)
and [FAO statistical database terms](https://www.fao.org/contact-us/terms/en/) before redistributing raw data.
The repository's code license does not relicense NOAA, CRU, FAO, or World Bank source material.

Starting context: [Fortune's June 2026 El Niño economic-impact article](https://fortune.com/2026/06/17/el-nino-global-economy-trillions-in-losses/).
