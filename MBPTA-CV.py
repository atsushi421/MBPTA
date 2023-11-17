import argparse
import os
import sys

import numpy as np
from scipy.stats import expon, kstest  # type: ignore
from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore

from src.config import (IID_TEST_THRESHOLD, MIN_REQUIRED_TAILS,  # type: ignore
                        NUM_IGNORE, PWCET_GRANULARITY, Z_VALUE)
from src.plot import plot_cv, plot_pwcet  # type: ignore


def ljung_box_test(et_array: np.ndarray) -> bool:
    return acorr_ljungbox(et_array, lags=20).iloc[-1]['lb_pvalue'] <= IID_TEST_THRESHOLD


def kolmogorov_smirnov_test(et_array: np.ndarray) -> bool:
    half_size = int(len(et_array) / 2)
    return kstest(et_array[:half_size], et_array[half_size:],
                  alternative='less').pvalue <= IID_TEST_THRESHOLD


def main(et_array: np.ndarray, title: str, plot_pwcet_flag: bool, plot_cv_flag: bool) -> None:
    half_size = int(len(et_array) / 2)
    et_array_sorted = np.sort(et_array)[::-1]

    # Calculate empirical residual CV
    residual_cv = np.zeros(half_size)
    for i in range(len(residual_cv)):
        threshold = et_array_sorted[i]
        excesses = et_array_sorted[:i + 1] - threshold
        residual_cv[i] = np.std(excesses) / np.mean(excesses)

    # Calculate acceptance region of Gumbel distribution (95% confidence interval)
    sqrt_vals = np.sqrt(np.arange(half_size))
    limit_above = 1 + Z_VALUE / sqrt_vals
    limit_below = 1 - Z_VALUE / sqrt_vals

    # (Optional) Plot CV and acceptance region
    if plot_cv_flag:
        plot_cv(
            residual_cv[NUM_IGNORE:],
            limit_above[NUM_IGNORE:],
            limit_below[NUM_IGNORE:],
            title
        )

    # Test exponentiality for 10 < N_{+th} <= 50
    for cv, la in zip(residual_cv[NUM_IGNORE:MIN_REQUIRED_TAILS+1],
                      limit_above[NUM_IGNORE:MIN_REQUIRED_TAILS+1]):
        if cv > la:
            print("[ERROR] Please increase the number of samples.")
            return

    # Find the best threshold
    best_cv = sys.maxsize
    num_tails = MIN_REQUIRED_TAILS
    for i in range(MIN_REQUIRED_TAILS+1, len(residual_cv)):
        if residual_cv[i] > limit_above[i]:  # Test exponentiality
            break
        if abs(residual_cv[i] - 1) < abs(best_cv - 1):
            best_cv = residual_cv[i]
            num_tails = i
    threshold = et_array_sorted[num_tails - 1]

    # Fitting
    excesses = et_array_sorted[:num_tails] - threshold
    lambda_ = 1 / np.mean(excesses)
    estimated_et_array = np.linspace(0, np.max(excesses) + threshold, PWCET_GRANULARITY)
    prob_cdf = expon.cdf(estimated_et_array, scale=1/lambda_)
    estimated_et_array += threshold
    pwcet_curve = np.column_stack((estimated_et_array, prob_cdf))
    pwcet_curve[0, 0] = 0

    # (Optional) Plot pWCET CDF
    if plot_pwcet_flag:
        plot_pwcet(pwcet_curve, et_array, title)

    print("[INFO] Analysis completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--latency_dir', default='./sample', help="Path to the latency directory."
    )
    parser.add_argument(
        '--plot_pwcet', action='store_true', default=True, help="Generate the pWCET plot if set."
    )
    parser.add_argument(
        '--plot_cv', action='store_true', default=True, help="Generate the CV plot if set."
    )
    args = parser.parse_args()

    for latency_file in os.listdir(args.latency_dir):
        if latency_file.endswith('.npy'):
            latencies = np.load(os.path.join(args.latency_dir, latency_file))
        elif latency_file.endswith('.txt'):
            with open(os.path.join(args.latency_dir, latency_file), 'r') as f:
                latencies = np.array(f.readlines(), dtype=int)
        else:
            raise NotImplementedError

        if len(latencies) < 100:
            print(f'[ERROR] Data size is less than 100: {latency_file}')
            continue
        if ljung_box_test(latencies):
            print(f'[ERROR] Ljung-Box test failed: {latency_file}')
            continue
        if kolmogorov_smirnov_test(latencies):
            print(f'[ERROR] Kolmogorov-Smirnov test failed: {latency_file}')
            continue

        main(
            latencies, f'{latency_file.split(".")[0]}', args.plot_pwcet, args.plot_cv
        )
