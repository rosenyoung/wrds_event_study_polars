# This is modified from Event-Study-Polars by Chihche-Liew
# https://github.com/Chihche-Liew/Event-Study-Polars

# Use local data instead of connecting to WRDS

# source code: # https://wrds-www.wharton.upenn.edu/pages/wrds-research/applications/python-replications/programming-python-wrds-event-study/
import os
import json
import polars as pl
import numpy as np
from statsmodels.api import OLS, add_constant
import duckdb as db
from polars import col
from tqdm import tqdm

# Modify paths as needed
input_dir = "D:/DataBase/wrds"

output_dir = "D:/DataBase/Research_Data/"

class EventStudy:
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

        estwins = estwin + gap + abs(evtwins)
        estwine = gap + abs(evtwins) + 1
        evtwins = abs(evtwins)
        evtrang = abs(evtwins) + evtwine + 1
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

        def process_grp(grp):
            grp = grp.sort('rdate')
            est = grp.filter(col('isest') == 1)
            evt = grp.filter(col('isevt') == 1)

            if est.height < minval or evt.height != evtrang or evt.filter(col('evtflag') == 1).height < 1:
                return pl.DataFrame()

            alpha = 0.0
            rmse = 1.0

            if model == 'madj':
                alpha = est.select(col('exret')).to_numpy().mean()
                rmse = est.select(col('exret')).to_numpy().std(ddof=1)
                evt = evt.with_columns([
                    pl.lit(alpha).alias('INTERCEPT'),
                    pl.lit(rmse).alias('RMSE'),
                    pl.lit(alpha).alias('alpha'),
                    col('exret').alias('abret'),
                    col('mkt').alias('expret')
                ])
            else:
                est_pd = est.to_pandas()
                y = est_pd['ret']
                if model == 'm':
                    X = add_constant(est_pd[['mktrf']])
                elif model == 'ff':
                    X = add_constant(est_pd[['mktrf','smb','hml']])
                elif model == 'ffm':
                    X = add_constant(est_pd[['mktrf','smb','hml','umd']])
                elif model == 'ff5':
                    X = add_constant(est_pd[['mktrf','smbf5','hmlf5','rmw','cma','umdf5']])
                res = OLS(y, X).fit()
                params = res.params.to_dict()
                alpha = params['const']
                betas = {k: v for k, v in params.items() if k != 'const'}
                rmse = np.sqrt(res.mse_resid)

                expr = pl.lit(alpha)
                for f_name, b_val in betas.items():
                    expr = expr + pl.lit(b_val) * col(f_name)
                evt = evt.with_columns([
                    pl.lit(alpha).alias('INTERCEPT'),
                    pl.lit(rmse).alias('RMSE'),
                    pl.lit(alpha).alias('alpha'),
                    expr.alias('expret'),
                    (col('ret') - expr).alias('abret')
                ])

            def compute_cret(ret_series):
                result = []
                acc = 0.0
                for r in ret_series:
                    tmp = (r * acc) + (r + acc)
                    acc = tmp
                    result.append(tmp)
                return pl.Series(result)

            def compute_cexpret(expret_series):
                result = []
                acc = 0.0
                for r in expret_series:
                    tmp = (r * acc) + (r + acc)
                    acc = tmp
                    result.append(tmp)
                return pl.Series(result)

            evt = evt.sort('date')
            evt = evt.with_columns([
                compute_cret(evt['ret']).alias('cret'),
                compute_cexpret(evt['expret']).alias('cexpret'),
                col('abret').cum_sum().alias('car'),
                pl.lit((est.height - 2) / (est.height - 4)).alias('pat_scale')
            ])

            evt = evt.with_columns([
                (col('abret') / rmse).cum_sum().alias('sar'),
                (col('cret') - col('cexpret')).alias('bhar'),
                (col('car') / np.sqrt(evtrang * rmse**2)).cum_sum().alias('scar')
            ])
            return evt

        keys = df.select(['permno','edate']).unique().to_pandas().to_dict(orient='records')
        result_list = []
        for evt in tqdm(keys):
            grp = df.filter((col('permno')==evt['permno']) & (col('edate')==evt['edate']))
            out = process_grp(grp)
            if out.height > 0:
                result_list.append(out)
        processed = pl.concat(result_list) if result_list else pl.DataFrame()
        df_evt = processed.drop_nulls().to_pandas()
        
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