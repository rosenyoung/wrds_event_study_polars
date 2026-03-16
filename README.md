# WRDS Event Study Polars

This project is a modified and extended version of
[Chihche-Liew/Event-Study-Polars](https://github.com/Chihche-Liew/Event-Study-Polars).

Several minor issues in the original implementation have been fixed, and additional features have been added.

**Update**: The previous version still transforms Polars DataFrames to Pandas and uses statsmodels to run OLS regressions. This could somewhat lower the speed.

To further improve the speed, I wanted to use [polar-ols](https://github.com/azmyrajab/polars_ols) to do the coefficient estimation in event study. I asked the Claude Code to read the previous version, describe my needs, and it generates the CLAUDE.md and the AI Prompt.md for me.

Then I asked Claude to carefully read the AI Prompt and compelete the task described in it. Claude finished the task in about 10 minutes, with some minor bugs. After a round of revision, Claude delivered the useful version of wrds_event_study_polars_ols, which is much more efficient than the previous version. The whole process was done in half an hour, while it might take more than half a day before.

## Overview

This package allows users to run **event studies using locally stored data in Parquet format**, rather than querying WRDS directly. It also supports **Fama–French 5-factor models**.

In local testing (Intel i9-12900K, 64GB RAM), the package completes an event study with approximately **10,000 events in about 71 seconds**, compared to roughly **7 minutes when running the same task on WRDS**.

**Update**: The new wrds_event_study_polars_ols can complete the example event study (about 1600 events) in 1 second, the old version of wrds_event_study_polars can finish it in 8 seconds, while it take more than 1 minute on WRDS.

## Requirements

pip install polars pandas numpy statsmodels duckdb tqdm polars-ols

## Data Preparation

To download WRDS data and store it locally as Parquet files, I recommend using:

- [iangow/db2pq](https://github.com/iangow/db2pq)

Data tables required to run the event study:

- crsp.dsf
- crsp.dsi
- crsp.dsedelist
- ff.factors_daily
- ff.fivefactors_daily

## How to Use

### Step 1: Prepare Event Data

Prepare a **Polars DataFrame** containing event dates with the following columns:

- `permno`
- `edate`

### Step 2: Modify Input Directory

Specify an input directory in the code to read the data.

### Step 3: Run the Event Study

```python
from wrds_event_study_polars import EventStudy

es = EventStudy()

results = es.eventstudy(
    data=evt_date,
    model='ffm',   # options: 'madj', 'm', 'ff', 'ffm', 'ff5', 'ff5m'
    estwin=250,
    evtwins=-10,
    evtwine=10,
    gap=10
)

event_stats = results['event_stats']
event_window = results['event_window']
event_date = results['event_date']
```

### Practical Tips

If the number of events exceeds 100,000, running the event study in chunks is recommended to avoid out-of-memory issues when querying data.
