import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from scipy.stats import expon  # type: ignore

from config import MIN_VALUE  # type: ignore

OUTPUT_DIR = '../output/'


def plot_cv(cv, limit_above, limit_below, title) -> None:
    plt.figure(figsize=(10, 7.5))
    plt.plot(cv[MIN_VALUE:], 'b-', linewidth=2)
    plt.plot(range(len(limit_above)), limit_above, 'r-', linewidth=2)
    plt.plot(range(len(limit_below)), limit_below, 'r-', linewidth=2)

    plt.xlim(0, len(cv))
    plt.ylim(0.3, 1.7)
    plt.title(title, fontsize=16)
    plt.savefig(f'{OUTPUT_DIR}/{title}_cv.png')
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
    plt.savefig(f'{OUTPUT_DIR}/{title}_pwcet_cdf.png')
