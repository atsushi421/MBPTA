import matplotlib.pyplot as plt  # type: ignore
import numpy as np

OUTPUT_DIR = './output/'


def plot_cv(residual_cv: np.ndarray, limit_above: np.ndarray, limit_below: np.ndarray, title: str) -> None:
    plt.figure(figsize=(10, 7.5))
    plt.plot(residual_cv[::-1], 'b-', linewidth=2)
    plt.plot(range(len(limit_above)), limit_above[::-1], 'r--', linewidth=2)
    plt.plot(range(len(limit_below)), limit_below[::-1], 'r--', linewidth=2)

    plt.xlim(-5, len(residual_cv)+5)
    plt.ylim(0.3, 1.7)
    plt.title(title, fontsize=16)
    plt.xlabel('Rejected Samples', fontsize=14)
    plt.ylabel('CV', fontsize=14)
    plt.savefig(f'{OUTPUT_DIR}/{title}_cv.png')
    plt.close()


def plot_pwcet(pwcet_curve: np.ndarray, et_array: np.ndarray, title: str) -> None:
    # Calculating the profiled CDF
    et_array = np.sort(et_array)
    prob_cdf = np.linspace(1/len(et_array), 1, len(et_array))
    et_curve = np.column_stack((et_array, prob_cdf))

    # Plotting the CDF
    plt.figure(figsize=(10, 6))
    plt.plot(pwcet_curve[:, 0], pwcet_curve[:, 1], 'b-', linewidth=3, label='Analyzed CDF')
    plt.plot(et_curve[:, 0], et_curve[:, 1], 'g--', linewidth=3, label='Profiled CDF')
    plt.title(title, fontsize=16)
    plt.xlabel('Execution Time', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{OUTPUT_DIR}/{title}_pwcet_cdf.png')
