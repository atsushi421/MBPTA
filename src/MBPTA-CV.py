import argparse
import os

import numpy as np
from scipy.stats import kstest  # type: ignore
from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore

from config import (IID_TEST_THRESHOLD, MIN_CV, MIN_VALUE,  # type: ignore
                    Z_VALUE)
from plot import plot_cv, plot_pwcet  # type: ignore


def ljung_box_test(et_array):
    return acorr_ljungbox(et_array, lags=20).iloc[-1]['lb_pvalue'] <= IID_TEST_THRESHOLD


def kolmogorov_smirnov_test(et_array):
    half_size = int(len(et_array) / 2)
    return kstest(et_array[:half_size], et_array[half_size:],
                  alternative='less').pvalue <= IID_TEST_THRESHOLD


def main(et_array, title: str, plot_pwcet_flag: bool, plot_cv_flag: bool) -> None:
    half_size = int(len(et_array) / 2)
    et_array_sorted = np.sort(et_array)[::-1]

    # Calculate CV
    cv = np.zeros(half_size)
    for i in range(half_size, MIN_VALUE, -1):
        slice_ = et_array_sorted[:i] - et_array_sorted[i-1]
        cv[MIN_VALUE + half_size - i] = np.sqrt(np.var(slice_)) / np.mean(slice_)

    # Calculate confidence interval
    limit_above = np.zeros(half_size - MIN_VALUE)
    limit_below = np.zeros(half_size - MIN_VALUE)
    for i in range(half_size - MIN_VALUE):
        sqrt_val = np.sqrt(half_size - i)
        limit_above[i] = 1 + Z_VALUE / sqrt_val
        limit_below[i] = 1 - Z_VALUE / sqrt_val

    if plot_cv_flag:
        plot_cv(cv, limit_above, limit_below, title)

    start_pos = 0
    for i in range(half_size - MIN_VALUE):
        if cv[MIN_VALUE + i] > limit_above[i]:
            start_pos = i

    if start_pos > (half_size - MIN_CV):
        print("[ERROR] Analysis not completed due to insufficient data")
        return

    best_cv = 1000.0
    num_elements = 0
    for i in range(start_pos, half_size - MIN_CV):
        if (abs(cv[MIN_VALUE + i] - 1) < abs(best_cv - 1)) and (cv[MIN_VALUE + i] < limit_above[i]):
            best_cv = cv[MIN_VALUE + i]
            num_elements = half_size - i

    # cv(th) exponentiality test
    for i in range(half_size - MIN_CV, half_size - MIN_VALUE):
        if cv[MIN_VALUE + i] > limit_above[i]:
            print("[ERROR] Analysis not completed due to insufficient data")
            return

    if plot_pwcet_flag:
        plot_pwcet(et_array, et_array_sorted, num_elements, title)

    print("[INFO] Analysis completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--latency_cache_dir', type=str, help='Path to the latency cache directory',
        default='~/autoware_optimization_tools/latency_cache'
    )
    parser.add_argument(
        '--plot_pwcet', action='store_true',
        help='Generate the pWCET plot if set', default=True
    )
    parser.add_argument(
        '--plot_cv', action='store_true',
        help='Generate the CV plot if set', default=False
    )
    args = parser.parse_args()

    for cache_file in os.listdir(args.latency_cache_dir):
        latencies = np.ceil(
            np.load(os.path.join(args.latency_cache_dir, cache_file)) * 10**(-3)
        ).astype(int)  # ns -> us

        if len(latencies) < 100:
            print(f'[ERROR] Data size is less than 100: {cache_file}')
            continue
        # if ljung_box_test(latencies):
        #     print(f'[ERROR] Ljung-Box test failed: {cache_file}')
        #     continue
        # if kolmogorov_smirnov_test(latencies):
        #     print(f'[ERROR] Kolmogorov-Smirnov test failed: {cache_file}')
        #     continue

        main(
            latencies, f'{cache_file.split(".")[0]}', args.plot_pwcet, args.plot_cv
        )
