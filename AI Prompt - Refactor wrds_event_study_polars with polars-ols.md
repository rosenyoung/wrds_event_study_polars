---
date: 2026-03-17
tags: #prompt #automation #polars #event-study
---
# AI Prompt: Replace statsmodels OLS with polars-ols in wrds_event_study_polars

## Context

I have a project called **wrds_event_study_polars** (file: wrds_event_study_polars.py). It is an event study framework that queries CRSP and Fama-French local data, then estimates factor models (market-adjusted, CAPM, Fama-French 3-factor, Carhart 4-factor) over an estimation window to compute abnormal returns, CARs, and BHARs.

The project already uses **Polars** and **DuckDB** for data manipulation and achieves ~240x speedup over the original WRDS version on ~12,000 events. However, there is a major bottleneck: **for the OLS regression step, it still converts Polars DataFrames to Pandas DataFrames and uses `statsmodels` to run OLS**. This Polars → Pandas → statsmodels → Polars round-trip is slow and defeats the purpose of using Polars for the rest of the pipeline.

I want you to **replace all statsmodels OLS regression calls with `polars-ols`**, a Rust-backed Polars extension that performs least squares natively inside the Polars engine, eliminating the Pandas conversion entirely.

---

## Your Task

Refactor wrds_event_study_polars.py so that **all OLS / factor-model coefficient estimation** is done via `polars-ols` instead of `statsmodels`.
Do not modify the wrds_event_study_polars.py directly, make a copy of the file and make modification on the copy.

Specifically:

### 1. Identify all regression entry points

Search the codebase for every place where:

- A Polars DataFrame is converted to Pandas (`.to_pandas()`)
- `statsmodels` is imported or used (e.g., `import statsmodels`, `sm.OLS`, `sm.add_constant`, `.fit()`, `.params`)
- OLS coefficients are extracted and mapped back into Polars

These are the locations that need refactoring.

### 2. Understand the regression models used

The event study supports four models. Each regresses **excess stock return** on factor(s) during the **estimation window**:

| Model key | Name                     | Regression                                                              |
| --------- | ------------------------ | ----------------------------------------------------------------------- |
| `madj`  | Market-adjusted          | No regression needed — abnormal return = stock return − market return |
| `m`     | Market model (CAPM-like) | `ret_excess ~ mktrf`                                                  |
| `ff`    | Fama-French 3-factor     | `ret_excess ~ mktrf + smb + hml`                                      |
| `ffm`   | Carhart 4-factor         | `ret_excess ~ mktrf + smb + hml + umd`                                |

For each event, the regression is estimated over the **estimation window** (e.g., 250 trading days before the gap). The estimated coefficients (alpha + betas) are then used to compute **expected returns** during the **event window**, and the difference gives the **abnormal return**.

### 3. Replace with polars-ols

Use the `polars-ols` package to perform all regressions natively in Polars. Here is the key reference for how to use it:

#### Installation

```
pip install polars-ols
```

#### Basic usage

```python
import polars as pl
import polars_ols as pls  # registers 'least_squares' namespace
```

#### Getting coefficients (what you need for event study)

```python
# Mode "coefficients" returns a Polars Struct with coefficient values
coefficients = df.select(
    pl.col("y").least_squares.ols(
        pl.col("x1"), pl.col("x2"),  # pass feature columns as positional args
        mode="coefficients",
        add_intercept=True
    ).alias("coefficients")
)
# Output: struct with fields {x1: beta1, x2: beta2, const: alpha}
```

#### Per-group coefficients (critical — one regression per event)

```python
# Use .over("group") to estimate separate regressions per group
coefficients_per_group = df.select(
    "group",
    pl.col("y").least_squares.ols(
        pl.col("x1"), pl.col("x2"),
        mode="coefficients",
        add_intercept=True
    ).over("group").alias("coefficients")
).unique(maintain_order=True)
```

#### Formula API alternative

```python
# Equivalent using formula syntax (like statsmodels)
coefficients = df.select(
    pl.col("y").least_squares.from_formula(
        "x1 + x2",  # RHS of formula
        mode="coefficients"
    ).alias("coefficients")
)
# Formula API adds intercept by default; use "x1 + x2 - 1" to suppress it
```

#### Getting predictions directly

```python
# Mode "predictions" returns fitted values (same length as input)
predictions = df.with_columns(
    pl.col("y").least_squares.ols(
        pl.col("x1"), pl.col("x2"),
        mode="predictions",
        add_intercept=True
    ).over("group").alias("expected_return")
)
```

#### Out-of-sample prediction

```python
# If you have coefficients from estimation window and want to predict on event window:
df_event.select(
    pl.col("coefficients_train").least_squares.predict(
        pl.col("mktrf"), pl.col("smb"), pl.col("hml")
    ).alias("expected_return")
)
```

#### Statistics output (optional, for diagnostics)

```python
statistics = df.select(
    pl.col("y").least_squares.ols(
        pl.col("x1"), pl.col("x2"),
        mode="statistics",
        add_intercept=True
    )
).unnest("statistics").explode(["feature_names", "coefficients", "standard_errors", "t_values", "p_values"])
```

### 4. Detailed refactoring plan

Here is the expected flow for each factor model (`m`, `ff`, `ffm`):

#### A. Estimation phase (estimation window data, per event)

**Before (statsmodels pattern):**

```python
# Typical existing pattern (pseudocode)
for each event (or grouped):
    est_df = polars_df.filter(estimation_window).to_pandas()
    X = sm.add_constant(est_df[['mktrf', 'smb', 'hml']])
    y = est_df['ret_excess']
    model = sm.OLS(y, X).fit()
    alpha, beta_mkt, beta_smb, beta_hml = model.params
```

**After (polars-ols pattern):**

```python
# All events at once, parallelized via .over("event_id")
est_data = polars_df.filter(pl.col("is_estimation_window"))

coefficients = est_data.select(
    "event_id",
    pl.col("ret_excess").least_squares.ols(
        pl.col("mktrf"), pl.col("smb"), pl.col("hml"),
        mode="coefficients",
        add_intercept=True
    ).over("event_id").alias("coefficients")
).unique(subset=["event_id"], maintain_order=True)
```

#### B. Prediction phase (event window data)

**Before:**

```python
expected_ret = alpha + beta_mkt * mktrf + beta_smb * smb + beta_hml * hml
abret = ret - expected_ret
```

**After (option 1 — manual dot product with unnested coefficients):**

```python
# Join coefficients back to event window data
event_data = polars_df.filter(pl.col("is_event_window")).join(coefficients, on="event_id")

# Unnest struct and compute expected return
event_data = event_data.unnest("coefficients").with_columns(
    (
        pl.col("const")
        + pl.col("mktrf_coef") * pl.col("mktrf")
        + pl.col("smb_coef") * pl.col("smb")
        + pl.col("hml_coef") * pl.col("hml")
    ).alias("expected_ret")
).with_columns(
    (pl.col("ret") - pl.col("expected_ret")).alias("abret")
)
```

**After (option 2 — use polars-ols predict):**

```python
# If the estimation & event window data can be structured together,
# use least_squares.predict with pre-computed coefficients
event_data = event_data.with_columns(
    pl.col("coefficients").least_squares.predict(
        pl.col("mktrf"), pl.col("smb"), pl.col("hml")
    ).alias("expected_ret")
)
```

### 5. Requirements & constraints

- **Do NOT change** any SQL queries, data loading, or WRDS connection logic.
- **Do NOT change** the output API — `es.eventstudy(...)` should still return the same dictionary structure (`event_stats`, `event_window`, `event_date`).
- **Do NOT change** the `madj` (market-adjusted) model — it doesn't use regression.
- **Remove** `statsmodels` from imports and dependencies after refactoring. Replace with `polars-ols` in the dependency list.
- **Preserve** numerical accuracy — the coefficients from `polars-ols` should match `statsmodels` OLS output (both use standard OLS; verify on a small test case).
- **Leverage parallelism** — use `.over("event_id")` or equivalent group-by mechanism so all events are estimated in a single vectorized pass (no Python loops over events).
- Update the dependency list: replace `statsmodels` with `polars-ols` in the README / install instructions.

### 6. Expected performance improvement

Based on `polars-ols` benchmarks:

- OLS: **2-3x faster** than numpy lstsq for single regression
- WLS: **4-6x faster** than statsmodels
- The main gain is **eliminating the Polars→Pandas→statsmodels→Polars conversion overhead** and **replacing Python loops with Rust-parallelized `.over()` group-by** operations
- For 12,000 events × 250-day estimation windows, this should yield a significant additional speedup on top of the existing 240x

### 7. Testing

After refactoring:

1. Run the event study with a small sample (e.g., 10 events) using each model (`m`, `ff`, `ffm`).
2. Compare the output (`abret`, `car`, `bhar`) against the old statsmodels-based version.
3. Differences should be within floating-point tolerance (< 1e-10).
4. Benchmark the full 12,000-event run to measure the speedup.


## References

Here is the references of the polars-ols

https://github.com/azmyrajab/polars_ols

and here is the README document



# Polars OLS

## Least squares extension in Polars

Supports linear model estimation in [Polars](https://www.pola.rs/).

This package provides efficient rust implementations of common linear
regression variants (OLS, WLS, Ridge, Elastic Net, Non-negative least squares, Recursive least squares) and exposes
them as simple polars expressions which can easily be integrated into your workflow.

### Why?

1. **High Performance**: implementations are written in rust and make use of optimized rust linear-algebra crates & LAPACK routines. See [benchmark](#benchmark) section.
2. **Polars Integration**: avoids unnecessary conversions from lazy to eager mode and to external libraries (e.g. numpy, sklearn) to do simple linear regressions.
   Chain least squares formulae like any other expression in polars.
3. **Efficient Implementations**:
   - Numerically stable algorithms are chosen where appropriate (e.g. SVD, QR).
   - Flexible model specification allows arbitrary combination of sample weighting, L1/L2 regularization, & non-negativity constraints on parameters.
   - Efficient rank-1 update algorithms used for moving window regressions.
4. **Easy Parallelism**: Computing OLS predictions, in parallel, across groups can not be easier: call `.over()` or `group_by` just like any other polars' expression and benefit from full Rust parallelism.
5. **Formula API**: supports building models via patsy syntax: `y ~ x1 + x2 + x3:x4 -1` (like statsmodels) which automatically converts to equivalent polars expressions.

Installation
------------

First, you need to [install Polars](https://pola-rs.github.io/polars/user-guide/installation/). Then run the below to install the `polars-ols` extension:

```console
pip install polars-ols
```

API & Examples
--------------

Importing `polars_ols` will register the namespace `least_squares` provided by this package.
You can build models either by either specifying polars expressions (e.g. `pl.col(...)`) for your targets and features or using
the formula api (patsy syntax). All models support the following general (optional) arguments:

- `mode` - a literal which determines the type of output produced by the model
- `null_policy` - a literal which determines how to deal with missing data
- `add_intercept` - a boolean specifying if an intercept feature should be added to the features
- `sample_weights` - a column or expression providing non-negative weights applied to the samples

Remaining parameters are model specific, for example `alpha` penalty parameter used by regularized least squares models.

See below for basic usage examples.
Please refer to the [tests](./tests/test_ols.py) or [demo notebook](./notebooks/polars_ols_demo.ipynb) for detailed examples.

```python
import polars as pl
import polars_ols as pls  # registers 'least_squares' namespace

df = pl.DataFrame({"y": [1.16, -2.16, -1.57, 0.21, 0.22, 1.6, -2.11, -2.92, -0.86, 0.47],
                   "x1": [0.72, -2.43, -0.63, 0.05, -0.07, 0.65, -0.02, -1.64, -0.92, -0.27],
                   "x2": [0.24, 0.18, -0.95, 0.23, 0.44, 1.01, -2.08, -1.36, 0.01, 0.75],
                   "group": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                   "weights": [0.34, 0.97, 0.39, 0.8, 0.57, 0.41, 0.19, 0.87, 0.06, 0.34],
                   })

lasso_expr = pl.col("y").least_squares.lasso("x1", "x2", alpha=0.0001, add_intercept=True).over("group")
wls_expr = pls.compute_least_squares_from_formula("y ~ x1 + x2 -1", sample_weights=pl.col("weights"))

predictions = df.with_columns(lasso_expr.round(2).alias("predictions_lasso"),
                              wls_expr.round(2).alias("predictions_wls"))

print(predictions.head(5))
```

```
shape: (5, 7)
┌───────┬───────┬───────┬───────┬─────────┬───────────────────┬─────────────────┐
│ y     ┆ x1    ┆ x2    ┆ group ┆ weights ┆ predictions_lasso ┆ predictions_wls │
│ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---     ┆ ---               ┆ ---             │
│ f64   ┆ f64   ┆ f64   ┆ i64   ┆ f64     ┆ f64               ┆ f64             │
╞═══════╪═══════╪═══════╪═══════╪═════════╪═══════════════════╪═════════════════╡
│ 1.16  ┆ 0.72  ┆ 0.24  ┆ 1     ┆ 0.34    ┆ 0.97              ┆ 0.93            │
│ -2.16 ┆ -2.43 ┆ 0.18  ┆ 1     ┆ 0.97    ┆ -2.23             ┆ -2.18           │
│ -1.57 ┆ -0.63 ┆ -0.95 ┆ 1     ┆ 0.39    ┆ -1.54             ┆ -1.54           │
│ 0.21  ┆ 0.05  ┆ 0.23  ┆ 1     ┆ 0.8     ┆ 0.29              ┆ 0.27            │
│ 0.22  ┆ -0.07 ┆ 0.44  ┆ 1     ┆ 0.57    ┆ 0.37              ┆ 0.36            │
└───────┴───────┴───────┴───────┴─────────┴───────────────────┴─────────────────┘
```

The `mode` parameter is used to set the type of the output returned by all methods (`"predictions", "residuals", "coefficients", "statistics"`).
It defaults to returning predictions matching the input's length.
Note that `"statistics"` is currently only supported for OLS/WLS/Ridge models.

In case `"coefficients"` is set the output is a [polars Struct](https://docs.pola.rs/user-guide/expressions/structs/) with coefficients as values and feature names as fields.
It's output shape 'broadcasts' depending on context, see below:

```python
coefficients = df.select(pl.col("y").least_squares.from_formula("x1 + x2", mode="coefficients")
                         .alias("coefficients"))

coefficients_group = df.select("group", pl.col("y").least_squares.from_formula("x1 + x2", mode="coefficients").over("group")
                        .alias("coefficients_group")).unique(maintain_order=True)

print(coefficients)
print(coefficients_group)
```

```
shape: (1, 1)
┌──────────────────────────────┐
│ coefficients                 │
│ ---                          │
│ struct[3]                    │
╞══════════════════════════════╡
│ {0.977375,0.987413,0.000757} │  # <--- coef for x1, x2, and intercept added by formula API
└──────────────────────────────┘
shape: (2, 2)
┌───────┬───────────────────────────────┐
│ group ┆ coefficients_group            │
│ ---   ┆ ---                           │
│ i64   ┆ struct[3]                     │
╞═══════╪═══════════════════════════════╡
│ 1     ┆ {0.995157,0.977495,0.014344}  │
│ 2     ┆ {0.939217,0.997441,-0.017599} │  # <--- (unique) coefficients per group
└───────┴───────────────────────────────┘
```

For dynamic models (like `rolling_ols`) or if in a `.over`, `.group_by`, or `.with_columns` context, the
coefficients will take the shape of the data it is applied on. For example:

```python
coefficients = df.with_columns(pl.col("y").least_squares.rls(pl.col("x1"), pl.col("x2"), mode="coefficients")
                         .over("group").alias("coefficients"))

print(coefficients.head())
```

```
shape: (5, 6)
┌───────┬───────┬───────┬───────┬─────────┬─────────────────────┐
│ y     ┆ x1    ┆ x2    ┆ group ┆ weights ┆ coefficients        │
│ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---     ┆ ---                 │
│ f64   ┆ f64   ┆ f64   ┆ i64   ┆ f64     ┆ struct[2]           │
╞═══════╪═══════╪═══════╪═══════╪═════════╪═════════════════════╡
│ 1.16  ┆ 0.72  ┆ 0.24  ┆ 1     ┆ 0.34    ┆ {1.235503,0.411834} │
│ -2.16 ┆ -2.43 ┆ 0.18  ┆ 1     ┆ 0.97    ┆ {0.963515,0.760769} │
│ -1.57 ┆ -0.63 ┆ -0.95 ┆ 1     ┆ 0.39    ┆ {0.975484,0.966029} │
│ 0.21  ┆ 0.05  ┆ 0.23  ┆ 1     ┆ 0.8     ┆ {0.975657,0.953735} │
│ 0.22  ┆ -0.07 ┆ 0.44  ┆ 1     ┆ 0.57    ┆ {0.97898,0.909793}  │
└───────┴───────┴───────┴───────┴─────────┴─────────────────────┘
```

For plain OLS/WLS and Ridge models, support has been recently added for producing a simple statistical significance
report. It can be used as such:

```python
statistics = (df.select(
   pl.col("y").least_squares.ols(pl.col("x1", "x2"), mode="statistics", add_intercept=True)
)
.unnest("statistics")  # results stored in a nested series by default
.explode(["feature_names", "coefficients", "standard_errors", "t_values", "p_values"])
)

print(statistics)
```

```
shape: (3, 8)
┌─────────┬──────────┬─────────┬──────────────┬──────────────┬─────────────┬───────────┬───────────┐
│ r2      ┆ mae      ┆ mse     ┆ feature_name ┆ coefficients ┆ standard_er ┆ t_values  ┆ p_values  │
│ ---     ┆ ---      ┆ ---     ┆ s            ┆ ---          ┆ rors        ┆ ---       ┆ ---       │
│ f64     ┆ f64      ┆ f64     ┆ ---          ┆ f64          ┆ ---         ┆ f64       ┆ f64       │
│         ┆          ┆         ┆ str          ┆              ┆ f64         ┆           ┆           │
╞═════════╪══════════╪═════════╪══════════════╪══════════════╪═════════════╪═══════════╪═══════════╡
│ 0.99631 ┆ 0.061732 ┆ 0.00794 ┆ x1           ┆ 0.977375     ┆ 0.037286    ┆ 26.212765 ┆ 3.0095e-8 │
│ 0.99631 ┆ 0.061732 ┆ 0.00794 ┆ x2           ┆ 0.987413     ┆ 0.037321    ┆ 26.457169 ┆ 2.8218e-8 │
│ 0.99631 ┆ 0.061732 ┆ 0.00794 ┆ const        ┆ 0.000757     ┆ 0.037474    ┆ 0.02021   ┆ 0.98444   │
└─────────┴──────────┴─────────┴──────────────┴──────────────┴─────────────┴───────────┴───────────┘
```

Finally, for convenience, in order to compute out-of-sample predictions you can use:
``least_squares.{predict, predict_from_formula}``. This saves you the effort of un-nesting the coefficients and doing the dot product in
python and instead does this in Rust, as an expression. Usage is as follows:

```python
df_test.select(pl.col("coefficients_train").least_squares.predict(pl.col("x1"), pl.col("x2")).alias("predictions_test"))
```

Supported Models
----------------

Currently, this extension package supports the following variants:

- Ordinary Least Squares: ``least_squares.ols``
- Weighted Least Squares: ``least_squares.wls``
- Regularized Least Squares (Lasso / Ridge / Elastic Net) ``least_squares.{lasso, ridge, elastic_net}``
- Non-negative Least Squares: ``least_squares.nnls``
- Multi-target Least Squares:  ``least_squares.multi_target_ols``

As well as efficient implementations of moving window models:

- Recursive Least Squares: ``least_squares.rls``
- Rolling / Expanding Window OLS: ``least_squares.{rolling_ols, expanding_ols}``

An arbitrary combination of sample_weights, L1/L2 penalties, and non-negativity constraints can be specified with
the ``least_squares.from_formula`` and ``least_squares.least_squares`` entry-points.

Solve Methods
-------------

`polars-ols` provides a choice over multiple supported numerical approaches per model (via `solve_method` flag),
with implications on performance vs numerical accuracy. These choices are exposed to the user for full control,
however, if left unspecified the package will choose a reasonable default depending on context.

For example, if you know you are dealing with highly collinear data, with unregularized OLS model, you may want to
explicitly set `solve_method="svd"` so that the minimum norm solution is obtained.

Benchmark
---------

The usual caveats of benchmarks apply here, but the below should still be indicative of the
type of performance improvements to expect when using this package.

This benchmark was run on randomly generated data with [pyperf](https://github.com/psf/pyperf) on my Apple M2 Max macbook
(32GB RAM, MacOS Sonoma 14.2.1). See [benchmark.py](./tests/benchmark.py) for implementation.

`<a id="bennchmark"></a>`

### n_samples=2_000, n_features=5

| Model                   | polars_ols         | Python Benchmark   | Benchmark Type     | Speed-up vs Python Benchmark |
| ----------------------- | ------------------ | ------------------ | ------------------ | ---------------------------- |
| Least Squares (QR)      | 195 µs ± 6 µs   | 466 µs ± 104 µs | Numpy (QR)         | 2.4x                         |
| Least Squares (SVD)     | 247 µs ± 5 µs   | 395 µs ± 69 µs  | Numpy (SVD)        | 1.6x                         |
| Ridge (Cholesky)        | 171 µs ± 8 µs   | 1.02 ms ± 0.29 ms | Sklearn (Cholesky) | 5.9x                         |
| Ridge (SVD)             | 238 µs ± 7 µs   | 1.12 ms ± 0.41 ms | Sklearn (SVD)      | 4.7x                         |
| Weighted Least Squares  | 334 µs ± 13 µs  | 2.04 ms ± 0.22 ms | Statsmodels        | 6.1x                         |
| Elastic Net (CD)        | 227 µs ± 7 µs   | 1.18 ms ± 0.19 ms | Sklearn            | 5.2x                         |
| Recursive Least Squares | 1.12 ms ± 0.23 ms | 18.2 ms ± 1.6 ms  | Statsmodels        | 16.2x                        |
| Rolling Least Squares   | 1.99 ms ± 0.03 ms | 22.1 ms ± 0.2 ms  | Statsmodels        | 11.1x                        |

### n_samples=10_000, n_features=100

| Model                   | polars_ols         | Python Benchmark     | Benchmark Type     | Speed-up vs Python Benchmark |
| ----------------------- | ------------------ | -------------------- | ------------------ | ---------------------------- |
| Least Squares (QR)      | 17.6 ms ± 0.3 ms  | 44.4 ms ± 9.3 ms    | Numpy (QR)         | 2.5x                         |
| Least Squares (SVD)     | 23.8 ms ± 0.2 ms  | 26.6 ms ± 5.5 ms    | Numpy (SVD)        | 1.1x                         |
| Ridge (Cholesky)        | 5.36 ms ± 0.16 ms | 475 ms ± 71 ms      | Sklearn (Cholesky) | 88.7x                        |
| Ridge (SVD)             | 30.2 ms ± 0.4 ms  | 400 ms ± 48 ms      | Sklearn (SVD)      | 13.2x                        |
| Weighted Least Squares  | 18.8 ms ± 0.3 ms  | 80.4 ms ± 12.4 ms   | Statsmodels        | 4.3x                         |
| Elastic Net (CD)        | 22.7 ms ± 0.2 ms  | 138 ms ± 27 ms      | Sklearn            | 6.1x                         |
| Recursive Least Squares | 270 ms ± 53 ms    | 57.8 sec ± 43.7 sec | Statsmodels        | 1017.0x                      |
| Rolling Least Squares   | 371 ms ± 13 ms    | 4.41 sec ± 0.17 sec | Statsmodels        | 11.9x                        |

- Numpy's `lstsq` (uses divide-and-conquer SVD) is already a highly optimized call into LAPACK and so the scope for speed-up is relatively limited,
  and the same applies to simple approaches like directly solving normal equations with Cholesky.
- However, even in such problems `polars-ols` Rust implementations for matching numerical algorithms tend to outperform by ~2-3x
- More substantial speed-up is achieved for the more complex models by working entirely in rust
  and avoiding overhead from back and forth into python.
- Expect a large additional relative order-of-magnitude speed up to your workflow if it involved repeated re-estimation of models in
  (python) loops.

Credits & Related Projects
--------------------------

- Rust linear algebra libraries [faer](https://faer-rs.github.io/getting-started.html) and [ndarray](https://docs.rs/ndarray/latest/ndarray/) support the implementations provided by this extension package
- This package was templated around the very helpful: [polars-plugin-tutorial](https://marcogorelli.github.io/polars-plugins-tutorial/)
- The python package [patsy](https://patsy.readthedocs.io/en/latest/formulas.html) is used for (optionally) building models from formulae
- Please check out the extension package [polars-ds](https://github.com/abstractqqq/polars_ds_extension) for general data-science functionality in polars

Future Work / TODOs
-------------------

- Support generic types, in rust implementations, so that both f32 and f64 types are recognized. Right now data is cast to f64 prior to estimation
- Add docs explaining supported models, signatures, and API
