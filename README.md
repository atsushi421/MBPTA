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
[1] M. Alcon, H. Tabani, L. Kosmidis, E. Mezzetti, J. Abella and F. J. Cazorla, "Timing of Autonomous Driving Software: Problem Analysis and Prospects for Future Solutions," 2020 IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS), Sydney, NSW, Australia, 2020, pp. 267-280, doi: 10.1109/RTAS48715.2020.000-1.