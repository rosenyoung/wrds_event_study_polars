# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A financial event study framework that computes abnormal stock returns around corporate events using locally stored Parquet files (sourced from WRDS via [db2pq](https://github.com/iangow/db2pq)) instead of querying WRDS directly. Modified from [Chihche-Liew/Event-Study-Polars](https://github.com/Chihche-Liew/Event-Study-Polars).

## Commands

This project uses `uv` as the package manager (Python >= 3.14).

```bash
uv sync                   # Install dependencies
python event_study_test.py  # Run the example/test
```

## Data Requirements

Parquet files must exist at `input_dir` (hardcoded to `D:/DataBase/wrds` at the top of `wrds_event_study_polars.py`):

| File | Contents |
|------|----------|
| `crsp/dsf.parquet` | CRSP daily stock file (permno, date, ret, …) |
| `crsp/dsi.parquet` | CRSP daily stock index (market returns, trading days) |
| `crsp/dsedelist.parquet` | Delisting returns (dlret) |
| `ff/factors_daily.parquet` | FF3 factors (mktrf, smb, hml, rf, umd) |
| `ff/fivefactors_daily.parquet` | FF5 factors (mktrf, smb, hml, rmw, cma, umd) |

Modify `input_dir` and `output_dir` at the top of `wrds_event_study_polars.py` before use.

## Architecture

The entire implementation lives in a single file: **`wrds_event_study_polars.py`**.

### `EventStudy.eventstudy()` pipeline

1. **SQL phase (DuckDB)**: Registers the input events DataFrame as an in-memory table, then runs a single multi-table join across events, `crsp.dsf`, `crsp.dsi`, `ff.factors`, and `ff.fivefactors`. Window functions (LAG/LEAD) calculate trading-day offsets for estimation and event windows. Outputs one row per (permno, edate, trading day).

2. **Polars transformation**: Type casting and data preparation.

3. **Group regression loop (tqdm)**: Iterates over each `(permno, edate)` group. Splits rows into estimation window and event window. Fits OLS (via `statsmodels`) using the selected factor model, then computes abnormal returns, CAR, SAR, SCAR, and BHAR.

### Window parameter conventions

- `estwin`: Length of the estimation window in trading days (default 250)
- `gap`: Trading days between estimation window end and event window start (default 10)
- `evtwins` / `evtwine`: Event window bounds relative to event date (negative = before, positive = after)
- Windows can be fully before or fully after the event date — **do not take absolute values of `evtwins`/`evtwine`** in any window arithmetic

### Supported models

| Key | Factors |
|-----|---------|
| `madj` | Market-adjusted (mean/std only) |
| `m` | Market model (mktrf) |
| `ff` | Fama-French 3-factor (mktrf, smb, hml) |
| `ffm` | FF3 + momentum (mktrf, smb, hml, umd) |
| `ff5` | FF5 (mktrf, smbf5, hmlf5, rmw, cma) |
| `ff5m` | FF5 + momentum (mktrf, smbf5, hmlf5, rmw, cma, umdf5) |

Note: FF5 uses separate factor columns (`smbf5`, `hmlf5`, `umdf5`) to distinguish from FF3 factors (`smb`, `hml`, `umd`).

### Output

`eventstudy()` returns a dict of three Polars DataFrames:
- `event_stats` — aggregate statistics by event time (across all events)
- `event_window` — daily returns for each stock in the event window
- `event_date` — cumulative metrics (CAR, BHAR, etc.) at the end of the event window

### Performance note

For >100k events, chunk the input to avoid out-of-memory errors.
