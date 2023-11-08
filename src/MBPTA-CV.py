import argparse

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from scipy.stats import expon, kstest  # type: ignore
from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore

MIN_VALUE = 10  # TODO: rename
MIN_CV = 50  # TODO: rename
IID_TEST_THRESHOLD = 0.05
Z_VALUE = 1.96  # 95% confidence interval


def plot_cv(cv, limit_above, limit_below, title) -> None:
    plt.figure(figsize=(10, 7.5))
    plt.plot(cv[MIN_VALUE:], 'b-', linewidth=2)
    plt.plot(range(len(limit_above)), limit_above, 'r-', linewidth=2)
    plt.plot(range(len(limit_below)), limit_below, 'r-', linewidth=2)

    plt.xlim(0, len(cv))
    plt.ylim(0.3, 1.7)
    plt.title(title, fontsize=16)
    plt.savefig(f'{title}_cv.png')
    plt.close()


def plot_pwcet_original(et_array, et_array_sorted, num_elements, title) -> None:
    excesses = et_array_sorted[:num_elements] - et_array_sorted[num_elements - 1]
    rate = 1 / np.mean(excesses)
    rank = np.linspace(0, 20 * np.max(excesses), 1000000)
    prob_ccdf = 1 - expon.cdf(rank, scale=1/rate)
    rank += et_array_sorted[num_elements - 1]
    pwcet_curve = np.column_stack((rank, prob_ccdf))

    # Generating ECCDF data
    rankeccdf = np.linspace(1.0, 0.0, len(et_array))
    traceeccdf = np.sort(et_array)
    eccdf1 = np.column_stack((traceeccdf, rankeccdf))

    # Adjusting pWCETcurve's first value
    pwcet_curve[0, 0] = 0

    # Setting up the plot
    plt.figure(figsize=(10, 6.25))
    y_range = [1e-16, 1]
    x_range = [np.nanmin(et_array) * 0.9, 2500]

    # Plotting the curves
    plt.plot(pwcet_curve[:, 0], pwcet_curve[:, 1], 'k-', linewidth=4, label=title)
    plt.plot(eccdf1[:, 0], eccdf1[:, 1], 'r--', linewidth=3)

    # Adding horizontal lines for probability thresholds
    plt.axhline(y=1e-3, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=1e-6, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=1e-9, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=1e-12, color='black', linestyle='--', linewidth=1)

    plt.yscale('log')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.title(title, fontsize=16)
    plt.savefig(f'{title}_pWCET.png')
    plt.close()


def plot_pwcet(et_array, et_array_sorted, num_elements, title) -> None:
    # Calculating the pWCET curve
    excesses = et_array_sorted[:num_elements] - et_array_sorted[num_elements - 1]
    rate = 1 / np.mean(excesses)
    rank = np.linspace(0, 20 * np.max(excesses), 1000000)
    prob_cdf = expon.cdf(rank, scale=1/rate)
    rank += et_array_sorted[num_elements - 1]
    pwcet_curve = np.column_stack((rank, prob_cdf))
    pwcet_curve[0, 0] = 0

    # Calculating the profiled CDF
    et_cdf = np.sort(et_array)
    rank_cdf = np.linspace(1/len(et_array), 1, len(et_array))
    cdf = np.column_stack((et_cdf, rank_cdf))

    # Plotting the CDF
    plt.figure(figsize=(10, 6))
    plt.plot(pwcet_curve[:, 0], pwcet_curve[:, 1], 'b-', linewidth=3, label='Analyzed CDF')
    plt.plot(cdf[:, 0], cdf[:, 1], 'r--', linewidth=3, label='Profiled CDF')
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title}_pwcet_cdf.png')


def main(input_file: str, title: str, plot_pwcet_flag: bool, plot_cv_flag: bool) -> None:
    with open(input_file, 'r') as f:
        et_array = np.array([float(line) for line in f.readlines()], dtype=np.float64)
    half_size = int(len(et_array) / 2)

    # Ljung-Box test
    if (acorr_ljungbox(et_array, lags=20).iloc[-1]['lb_pvalue'] <= IID_TEST_THRESHOLD):
        print("Ljung-Box test failed")
        return

    # Kolmogorov-Smirnov Test
    if (kstest(et_array[:half_size], et_array[half_size:],
               alternative='less').pvalue <= IID_TEST_THRESHOLD):
        print("Kolmogorov-Smirnov test failed")
        return

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
        print("Analysis not completed due to insufficient data")
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
            print("Analysis not completed due to insufficient data")
            return

    if plot_pwcet_flag:
        plot_pwcet(et_array, et_array_sorted, num_elements, title)

    print("Analysis completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file', type=str, help='Path to the input file trace',
        default='/home/atsushi/MBPTA/sample/sample_input.txt'
    )
    parser.add_argument(
        '--title', type=str, help='Title for plots', default='Sample'
    )
    parser.add_argument(
        '--plot_pwcet', action='store_true',
        help='Generate the pWCET plot if set', default=True
    )
    parser.add_argument(
        '--plot_cv', action='store_true',
        help='Generate the CV plot if set', default=True
    )
    args = parser.parse_args()

    main(args.input_file, args.title, args.plot_pwcet, args.plot_cv)
