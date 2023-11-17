# MBPTA-CV

## Setup

```bash
$ python3 -m pip install -r requirements.txt
```

## Preparation
Please store the task's latency statistics to files in one of the following formats:
- `.txt`
- `.npy`

Example is available in the [sample directory](https://github.com/atsushi421/MBPTA/tree/main/sample).

## Usage

```bash
python3 MBPTA-CV.py [-h] [--latency_dir LATENCY_DIR] [--plot_pwcet] [--plot_cv]
```

```bash
options:
  -h, --help            show this help message and exit
  --latency_dir LATENCY_DIR
                        Path to the latency directory.
  --plot_pwcet          Generate the pWCET plot if set.
  --plot_cv             Generate the CV plot if set.
```

## Reference
Jaume Abella, Maria Padilla, Joan Del Castillo, and Francisco J. Cazorla. 2017. Measurement-Based Worst-Case Execution Time Estimation Using the Coefficient of Variation. ACM Trans. Des. Autom. Electron. Syst. 22, 4, Article 72 (October 2017), 29 pages. https://doi.org/10.1145/3065924

> [!TIP]
> A summary of this paper is provided [here](https://atsushi421.github.io/markdown_share/paper/2017_TODAES_Measurement-Based_Worst-Case_Execution_Time_Estimation_Using/summary_2017_TODAES_Measurement-Based_Worst-Case_Execution_Time_Estimation_Using/).