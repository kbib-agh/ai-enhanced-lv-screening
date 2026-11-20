import os
import time
from functools import partial
from multiprocessing import Pool
from typing import Callable

from django.conf import settings
from django.db import connections
from tqdm import tqdm

_DEBUG_MODE = bool('PYDEVD_LOAD_VALUES_ASYNC' in os.environ or 'PYDEVD_LOAD_VALUES_ON_DEMAND' in os.environ)


def _call_worker(args_and_index, worker_func):
    index, args = args_and_index
    return index, worker_func(*args)


def parallel(worker: Callable, parameters: list[tuple]):
    tic = time.perf_counter()
    worker_results = []
    if _DEBUG_MODE:
        for params in parameters:
            worker_results.append(worker(*params))
    else:
        connections.close_all()
        enumerated_results = []
        with Pool(settings.N_PROC) as pool:
            for result in tqdm(
                pool.imap_unordered(partial(_call_worker, worker_func=worker), enumerate(parameters)),
                total=len(parameters)
            ):
                enumerated_results.append(result)
        worker_results = [result for index, result in sorted(enumerated_results)]

    toc = time.perf_counter()
    print(f'Total time ({len(parameters)} parameters): {toc - tic:.2f}s')

    return worker_results
