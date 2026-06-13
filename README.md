# WRDS Event Study Polars

This repository runs event studies from locally stored WRDS Parquet files. It is a modified and extended version of [Chihche-Liew/Event-Study-Polars](https://github.com/Chihche-Liew/Event-Study-Polars), with support for local data, Fama-French factor models, and a faster `polars-ols` implementation.

The recommended implementation is `wrds_event_study_polars_ols.py`, which exposes `EventStudy_Polars`. It uses DuckDB for the local Parquet joins, Polars for data processing, and `polars-ols` for vectorized OLS estimation. The older `wrds_event_study_polars.py` module is kept for comparison and uses `statsmodels` in a per-event loop.

## Update

The previous version still transformed Polars DataFrames to Pandas and used `statsmodels` to run OLS regressions. This could lower the speed because each event had to be processed through a Python regression loop.

To further improve performance, I wanted to use [polars-ols](https://github.com/azmyrajab/polars_ols) for coefficient estimation in the event study. I asked Claude Code to read the previous version, describe my needs, and generate `CLAUDE.md` and the AI prompt document for this repository.

Then I asked Claude to carefully read the AI prompt and complete the task described in it. Claude finished the first implementation in about 10 minutes, with some minor bugs. After a round of revision, Claude delivered the useful `wrds_event_study_polars_ols.py` version, which is much more efficient than the previous implementation. The whole process took about half an hour, while it might have taken more than half a day before.

## Requirements

- Python 3.14, as specified by `.python-version` and `pyproject.toml`
- Local WRDS data exported to Parquet
- `uv` for environment and dependency management

Install or sync the environment from the repository root:

```bash
uv sync
```

If `uv` cannot find Python 3.14 locally, install it with uv first:

```bash
uv python install 3.14
uv sync
```

Run Python commands through uv so the locked environment is used:

```bash
uv run python event_study_test.py
```

## Data Preparation

This project does not query WRDS directly. Download the required WRDS tables first and save them as Parquet files. A recommended tool for this is [iangow/db2pq](https://github.com/iangow/db2pq).

The code expects this folder layout under `input_dir`:

```text
<input_dir>/
  crsp/
    dsf.parquet
    dsi.parquet
    dsedelist.parquet
  ff/
    factors_daily.parquet
    fivefactors_daily.parquet
```

Required tables:

| File | Purpose |
| --- | --- |
| `crsp/dsf.parquet` | CRSP daily stock returns, including `permno`, `date`, and `ret` |
| `crsp/dsi.parquet` | CRSP daily trading calendar used to build estimation and event windows |
| `crsp/dsedelist.parquet` | Delisting returns, including `permno`, `dlstdt`, and `dlret` |
| `ff/factors_daily.parquet` | Daily market, FF3, risk-free, and momentum factors |
| `ff/fivefactors_daily.parquet` | Daily FF5 factors and momentum |

Before running the study, edit the module-level paths near the top of the implementation file:

```python
# wrds_event_study_polars_ols.py
input_dir = "D:/DataBase/wrds"
output_dir = "D:/DataBase/Research_Data/"
```

Use the same edit in `wrds_event_study_polars.py` only if you want to run the older statsmodels version.

## Event Input

Prepare a Polars DataFrame with one row per event and these columns:

| Column | Type | Description |
| --- | --- | --- |
| `permno` | integer | CRSP PERMNO |
| `edate` | date | Event date |

The included `evt_date_2020.txt` sample is a space-separated file with no header:

```text
14987 2020-05-05
11803 2020-05-01
```

Load it like this:

```python
import polars as pl

evt_date = (
    pl.read_csv(
        "evt_date_2020.txt",
        separator=" ",
        has_header=False,
        new_columns=["permno", "edate"],
        schema_overrides={"permno": pl.Int64, "edate": pl.Utf8},
    )
    .with_columns(pl.col("edate").str.to_date("%Y-%m-%d"))
)
```

## Basic Usage

Run the recommended `polars-ols` version from the repository root:

```python
from wrds_event_study_polars_ols import EventStudy_Polars

es = EventStudy_Polars()

results = es.eventstudy(
    data=evt_date,
    model="ff5m",
    estwin=250,
    gap=10,
    evtwins=-10,
    evtwine=10,
    minval=100,
)

event_stats = results["event_stats"]
event_window = results["event_window"]
event_date = results["event_date"]
```

The returned objects are Pandas DataFrames:

| Result key | Contents |
| --- | --- |
| `event_stats` | Aggregate statistics by event time across valid events |
| `event_window` | Event-window daily returns and abnormal returns for each event |
| `event_date` | End-of-window cumulative metrics for each event |

To save the outputs:

```python
event_stats.to_csv("event_stats.csv", index=False)
event_window.to_csv("event_window.csv", index=False)
event_date.to_csv("event_date.csv", index=False)
```

## Model Choices

Pass one of these strings to `model`:

| Model | Description | Factors |
| --- | --- | --- |
| `madj` | Market-adjusted model | No regression factors |
| `m` | Market model | `mktrf` |
| `ff` | Fama-French 3-factor | `mktrf`, `smb`, `hml` |
| `ffm` | Fama-French 3-factor plus momentum | `mktrf`, `smb`, `hml`, `umd` |
| `ff5` | Fama-French 5-factor | `mktrf`, `smbf5`, `hmlf5`, `rmw`, `cma` |
| `ff5m` | Fama-French 5-factor plus momentum | `mktrf`, `smbf5`, `hmlf5`, `rmw`, `cma`, `umdf5` |

`ff5` and `ff5m` use renamed factor columns from `fivefactors_daily.parquet` to avoid collisions with the FF3 columns.

## Window Parameters

`eventstudy()` uses trading days, not calendar days.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `estwin` | `250` | Number of trading days in the estimation window |
| `gap` | `10` | Trading-day gap between the estimation window and event window |
| `evtwins` | `-10` | Event-window start relative to the event date |
| `evtwine` | `10` | Event-window end relative to the event date |
| `minval` | `100` | Minimum required estimation-window observations |

Examples:

```python
# Symmetric 21-day window around the event date
evtwins=-10
evtwine=10

# Post-event window only
evtwins=0
evtwine=10

# Window fully before the event date
evtwins=-20
evtwine=-1
```

`evtwins` must be less than or equal to `evtwine`.

## Running the Included Comparison Script

`event_study_test.py` loads `evt_date_2020.txt`, runs both implementations, and compares the CAR and BHAR outputs.

After editing `input_dir` in both implementation files:

```bash
uv run python event_study_test.py
```

This script is useful as a smoke test after changing data paths or dependencies. It requires the local WRDS Parquet files to be present.

## Performance Notes

In local testing, the `EventStudy_Polars` implementation is substantially faster than the older statsmodels loop because the regressions are vectorized through `polars-ols`.

For very large event sets, especially above 100,000 events, process events in batches to reduce peak memory use. Event-level outputs such as `event_date` and `event_window` can be concatenated after batch runs. If you need exact aggregate statistics across all events, recompute the aggregate statistics from the combined event-level data rather than averaging the per-batch `event_stats` tables.

## Troubleshooting

- Run commands from the repository root so local imports resolve.
- Make sure `input_dir` points to the parent folder that contains the `crsp` and `ff` subfolders.
- Ensure `edate` is a Polars `Date`, not a string.
- If a run returns empty tables, check that each event date is a CRSP trading day, the event has enough estimation-window observations, and the full event window is available.
- If Python or dependencies are missing, run `uv sync` again and then use `uv run python ...`.
