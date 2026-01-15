# WRDS Event Study (Local)

This project is a modified and extended version of  
[Chihche-Liew/Event-Study-Polars](https://github.com/Chihche-Liew/Event-Study-Polars).

Several minor issues in the original implementation have been fixed, and additional features have been added.

## Overview

This package allows users to run **event studies using locally stored data in Parquet format**, rather than querying WRDS directly. It also supports **Fama–French 5-factor models**.

In local testing (Intel i9-12900K, 64GB RAM), the package completes an event study with approximately **10,000 events in about 71 seconds**, compared to roughly **7 minutes when running the same task on WRDS**.

## Requirements

pip install polars pandas numpy statsmodels duckdb tqdm

## Data Preparation

To download WRDS data and store it locally as Parquet files, I recommend using:

- [iangow/db2pq](https://github.com/iangow/db2pq)

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
    model='ffm',   # options: 'madj', 'm', 'ff', 'ffm', 'ff5
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
