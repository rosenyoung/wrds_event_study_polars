# This is modified from Event-Study-Polars by Chihche-Liew
# https://github.com/Chihche-Liew/Event-Study-Polars

# Use local data instead of connecting to WRDS

# source code: # https://wrds-www.wharton.upenn.edu/pages/wrds-research/applications/python-replications/programming-python-wrds-event-study/

# This version replaces statsmodels OLS with polars-ols for vectorized regression,
# eliminating the Python loop over events and the Polars->Pandas->statsmodels round-trip.

import os
import polars as pl
import polars_ols as pls  # registers 'least_squares' namespace on pl.Expr
import duckdb as db
from polars import col
import warnings

# Modify paths as needed
input_dir = "D:/DataBase/wrds"

output_dir = "D:/DataBase/Research_Data/"

# Factor columns for each supported model
MODEL_FACTORS = {
    'madj': [],
    'm': ['mktrf'],
    'ff': ['mktrf', 'smb', 'hml'],
    'ffm': ['mktrf', 'smb', 'hml', 'umd'],
    'ff5': ['mktrf', 'smbf5', 'hmlf5', 'rmw', 'cma'],
    'ff5m': ['mktrf', 'smbf5', 'hmlf5', 'rmw', 'cma', 'umdf5'],
}

class EventStudy_Polars:
    def __init__(self, output_path=output_dir):
        self.output_path = output_path or os.path.expanduser('~')

    def connect(self):
        return db.connect(database=':memory:', read_only=False)

    def eventstudy(self,
                   data=None,
                   model='madj',
                   estwin=250,
                   gap=10,
                   evtwins=-10,
                   evtwine=10,
                   minval=100,
                   output='df'):
        # Do not use absolute values for evtwins and evtwine to support event windows fully before or after event date
        if evtwins > evtwine:
            raise ValueError("evtwins should be less than or equal to evtwine.")
        estwins = estwin + gap - evtwins
        estwine = gap - evtwins + 1
        evtrang = evtwine - evtwins + 1
        evtwins = -evtwins
        evtwinx = estwins + 1

        evts = data
        # Register events as a DuckDB table instead of using PostgreSQL json_to_recordset
        conn = self.connect()
        events_df = pl.DataFrame(evts)
        conn.register('events', events_df.to_pandas())

        # Register events as a DuckDB table instead of using PostgreSQL json_to_recordset, so drop the events parameter in the params
        params = {'estwins': estwins, 'estwine': estwine, 'evtwins': evtwins, 'evtwine': evtwine, 'evtwinx': evtwinx}

        sql = f"""
        SELECT
                a.*,
                x.*,
                c.date as rdate,
                c.ret as ret1,
                (f.mktrf+f.rf) as mkt,
                f.mktrf,
                f.rf,
                f.smb,
                f.hml,
                f.umd,
                f5.smb as smbf5, -- smb is different from ff3 in ff5
                f5.hml as hmlf5,
                f5.rmw,
                f5.cma,
                f5.umd as umdf5,
                (1+c.ret)*(coalesce(d.dlret,0.00)+1)-1-(f.mktrf+f.rf) as exret,
                (1+c.ret)*(coalesce(d.dlret,0.00)+1)-1 as ret,
                case when c.date between a.estwin1 and a.estwin2 then 1 else 0 end as isest,
                case when c.date between a.evtwin1 and a.evtwin2 then 1 else 0 end as isevt,
                case
                  when c.date between a.evtwin1 and a.evtwin2 then (rank() OVER (PARTITION BY x.evtid ORDER BY c.date)-%(evtwinx)s)
                  else (rank() OVER (PARTITION BY x.evtid ORDER BY c.date))
                end as evttime,
                case
                  when c.date = a.date then 1
                  else 0
                end as evtflag
        FROM
          (
            SELECT
              date,
              lag(date, %(estwins)s ) over (order by date) as estwin1,
              lag(date, %(estwine)s )  over (order by date) as estwin2,
              lag(date, %(evtwins)s )  over (order by date) as evtwin1,
              lead(date, %(evtwine)s )  over (order by date) as evtwin2
            FROM '{input_dir}/crsp/dsi.parquet'
          ) as a
        JOIN
            (select
                upper(strftime(cast(edate as DATE), '%%d%%b%%Y')) || permno as evtid,
                permno,
                cast(edate as DATE) as edate
            from
                events
            ) as x
          ON a.date=x.edate
        JOIN '{input_dir}/crsp/dsf.parquet' c
            ON x.permno=c.permno
            AND c.date BETWEEN a.estwin1 and a.evtwin2
        JOIN '{input_dir}/ff/factors_daily.parquet' f
            ON c.date=f.date
        JOIN '{input_dir}/ff/fivefactors_daily.parquet' f5
            ON c.date=f5.date
        LEFT JOIN '{input_dir}/crsp/dsedelist.parquet' d
            ON x.permno=d.permno
            AND c.date=d.dlstdt
        WHERE f.mktrf is not null
        AND f5.mktrf is not null
        AND c.ret is not null
        ORDER BY x.evtid, x.permno, a.date, c.date
        """ % params
        df = conn.execute(sql).pl()

        df = df.with_columns([
            col('edate').cast(pl.Date),
            col('rdate').cast(pl.Date),
            # Transform returns to numeric types if needed
            col('ret').cast(pl.Float64),
            col('exret').cast(pl.Float64),
            col('mkt').cast(pl.Float64),
            col('mktrf').cast(pl.Float64),
            col('rf').cast(pl.Float64),
            col('smb').cast(pl.Float64),
            col('hml').cast(pl.Float64),
            col('umd').cast(pl.Float64),
            col('smbf5').cast(pl.Float64),
            col('hmlf5').cast(pl.Float64),
            col('rmw').cast(pl.Float64),
            col('cma').cast(pl.Float64),
            col('umdf5').cast(pl.Float64)
        ])

        # --- Vectorized regression using polars-ols (no Python loop) ---

        factor_cols = MODEL_FACTORS[model]
        n_params = len(factor_cols) + 1  # +1 for intercept

        # Split into estimation and event windows
        est_data = df.filter(col('isest') == 1)
        evt_data = df.filter(col('isevt') == 1)

        # Filter valid events: enough estimation data and correct event window size
        est_counts = est_data.group_by('evtid').agg(pl.len().alias('est_count'))
        evt_counts = evt_data.group_by('evtid').agg(pl.len().alias('evt_count'))
        valid_events = (
            est_counts.filter(col('est_count') >= minval)
            .join(evt_counts.filter(col('evt_count') == evtrang), on='evtid')
        )

        est_data = est_data.join(valid_events.select('evtid'), on='evtid')
        evt_data = evt_data.join(valid_events.select('evtid', 'est_count'), on='evtid')

        if est_data.height == 0 or evt_data.height == 0:
            import pandas as pd
            empty_pd = pd.DataFrame()
            return {'event_stats': empty_pd, 'event_window': empty_pd, 'event_date': empty_pd}

        if model == 'madj':
            # Market-adjusted model: no regression needed
            group_stats = est_data.group_by('evtid').agg([
                col('exret').mean().alias('alpha'),
                col('exret').std().alias('RMSE')
            ])
            evt_data = evt_data.join(group_stats, on='evtid')
            evt_data = evt_data.with_columns([
                col('alpha').alias('INTERCEPT'),
                col('exret').alias('abret'),
                col('mkt').alias('expret')
            ])
        else:
            # Factor model regression using polars-ols (vectorized across all events)
            feature_exprs = [pl.col(c) for c in factor_cols]

            # Get per-event OLS coefficients in a single vectorized pass
            coefficients = est_data.select(
                "evtid",
                pl.col("ret").least_squares.ols(
                    *feature_exprs,
                    mode="coefficients",
                    add_intercept=True
                ).over("evtid").alias("coefficients")
            ).unique(subset=["evtid"], maintain_order=True)

            # Get per-event residuals for RMSE calculation
            est_with_resid = est_data.with_columns(
                pl.col("ret").least_squares.ols(
                    *feature_exprs,
                    mode="residuals",
                    add_intercept=True
                ).over("evtid").alias("residual")
            )

            rmse_df = est_with_resid.group_by('evtid').agg(
                (col('residual').pow(2).sum() / (pl.len() - n_params)).sqrt().alias('RMSE')
            )

            # Join coefficients and RMSE to event window data
            evt_data = (
                evt_data
                .join(coefficients, on='evtid')
                .join(rmse_df, on='evtid')
            )

            # Compute expected returns via manual dot product with coefficients struct
            expret_expr = col('coefficients').struct.field('const')
            for c in factor_cols:
                expret_expr = expret_expr + col('coefficients').struct.field(c) * col(c)

            evt_data = evt_data.with_columns([
                expret_expr.alias('expret'),
                col('coefficients').struct.field('const').alias('alpha'),
            ])
            evt_data = evt_data.with_columns([
                col('alpha').alias('INTERCEPT'),
                (col('ret') - col('expret')).alias('abret')
            ])

            # Drop struct column before pandas conversion
            evt_data = evt_data.drop('coefficients')

        # Sort by event and date for correct cumulative calculations
        evt_data = evt_data.sort(['evtid', 'rdate'])

        # Compute cumulative returns and abnormal return metrics
        evt_data = evt_data.with_columns([
            ((1 + col('ret')).cum_prod().over('evtid') - 1).alias('cret'),
            ((1 + col('expret')).cum_prod().over('evtid') - 1).alias('cexpret'),
            col('abret').cum_sum().over('evtid').alias('car'),
            ((col('est_count') - 2) / (col('est_count') - 4)).alias('pat_scale')
        ])

        evt_data = evt_data.with_columns([
            (col('abret') / col('RMSE')).cum_sum().over('evtid').alias('sar'),
            (col('cret') - col('cexpret')).alias('bhar'),
            (col('car') / (pl.lit(evtrang) * col('RMSE').pow(2)).sqrt()).cum_sum().over('evtid').alias('scar')
        ])

        # Convert to pandas for final stats query and output
        df_evt = evt_data.drop_nulls().to_pandas()

        df_stats = db.query(f"""
            SELECT evttime,
                   AVG(car) AS car_m,
                   AVG(ret) AS ret_m,
                   AVG(abret) AS abret_m,
                   STDDEV_SAMP(abret) AS abret_v,
                   AVG(sar) AS sar_m,
                   STDDEV_SAMP(sar) AS sar_v,
                   AVG(scar) AS scar_m,
                   STDDEV_SAMP(scar) AS scar_v,
                   AVG(bhar) AS bhar_m,
                   COUNT(*) AS n,
                   AVG(cret) AS cret_edate_m,
                   AVG(car) AS car_edate_m,
                   AVG(bhar) AS bhar_edate_m
            FROM df_evt
            GROUP BY evttime
            ORDER BY evttime
        """
        ).df()

        df_window = df_evt[['permno','edate','rdate','evttime','ret','abret']].sort_values(['permno','evttime'])
        max_t = df_evt['evttime'].max()
        df_date = df_evt[df_evt['evttime'] == max_t][['permno','edate','cret','car','bhar']]
        df_date = df_date.sort_values(['permno','edate'])

        if output == 'df':
            return {'event_stats': df_stats,
                    'event_window': df_window,
                    'event_date': df_date}
