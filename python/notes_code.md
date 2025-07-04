
### Comparison compute_daily-mean_dask.py and compute_daily-mean_dask1.py
For computing the daily mean on one year of data, using 4 cpus on tintin:
*_dask.py ~ 4 min 10 sec (uses dask with Client() cluster options)
*_mpc.py ~ 3 min 55 sec (uses starmap in multiprocessing package)
