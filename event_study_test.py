

import importlib
import wrds_event_study_polars
import wrds_event_study_polars_ols
importlib.reload(wrds_event_study_polars)
importlib.reload(wrds_event_study_polars_ols)
from wrds_event_study_polars import EventStudy
from wrds_event_study_polars_ols import EventStudy_Polars
import polars as pl


evt_date = pl.read_csv(
    'evt_date_2020.txt',
    separator=' ',
    has_header=False,
    new_columns=['permno', 'edate'],
    schema_overrides={'permno': pl.Int64, 'edate': pl.Utf8},
).with_columns(pl.col('edate').str.to_date('%Y-%m-%d'))

es = EventStudy()
results = es.eventstudy(data=evt_date,
                        model='ff5m',  # options: 'madj', 'm', 'ff', 'ffm', 'ff5', 'ff5m'
                        estwin=250,
                        evtwins=3,
                        evtwine=10,
                        gap=10)

cum_ret = results['event_date']
event_stats = results['event_stats']
event_window = results['event_window']

esp = EventStudy_Polars()

import time
start_time = time.time()
results_esp= esp.eventstudy(data=evt_date,
                        model='ff5m',  # options: 'madj', 'm', 'ff', 'ffm', 'ff5', 'ff5m'
                        estwin=250,
                        evtwins=3,
                        evtwine=10,
                        gap=10)

event_stats_esp = results_esp['event_stats']
cum_ret_esp = results_esp['event_date']
event_window_esp = results_esp['event_window']
end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")

# turn cum_ret and cum_ret_esp into polars DataFrames
cum_ret = pl.DataFrame(cum_ret)
cum_ret_esp = pl.DataFrame(cum_ret_esp)

# Sort cum_ret and cum_ret_esp by 'permno' and 'edate', then compare the car and bhr of both results
cum_ret_sorted = cum_ret.sort(['permno', 'edate'])
cum_ret_esp_sorted = cum_ret_esp.sort(['permno', 'edate'])

# Compare the car and bhr columns of both results
comparison = cum_ret_sorted.join(cum_ret_esp_sorted, on=['permno', 'edate'], how='inner', suffix='_esp')
# Check if the car and bhr columns are close enough (within a small tolerance)
tolerance = 1e-6
comparison = comparison.with_columns(
    ((pl.col('car') - pl.col('car_esp')).abs() < tolerance).alias('car_close'),
    ((pl.col('bhar') - pl.col('bhar_esp')).abs() < tolerance).alias('bhar_close'),
)
# Print the number of rows where car and bhr are not close
num_car_not_close = comparison.filter(~pl.col('car_close')).height
num_bhar_not_close = comparison.filter(~pl.col('bhar_close')).height

print(f"Number of rows where car is not close: {num_car_not_close}")
print(f"Number of rows where bhar is not close: {num_bhar_not_close}")

