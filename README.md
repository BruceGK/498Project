# Evaluate and Detect Attacks in Financial Market with Simulation
This repository contains codes for course project authored by Chenhao Huang, Shlok Khandelwal and Zizhen Lian for 
CMSC498P/798P.

In this project, we study the problem of evaluating and detecting adversarial attacks in financial market by 
using a simulation environment ([ABIDES](https://github.com/abides-sim/abides)).

## Framework/Pipeline
### Files/Folders   
- `config`: folder contains config scripts to use with ABIDES for running experiments with attack agents and 
detection methods.
    + `rmsc03_analysis.py`: 
- `ABIDES_to_Lobster_data.py`: 

### Setups
Before running the configs, make sure submodule `abides` is checked out and the environment variable `PYTHONPATH` 
contains the root of the repo `.` and `./abides`.

### Commands

Running the experiments
```bash
# usage
python config/rmsc03_analysis.py -c rmsc03 -t <ticker symbol> -d <historical date> -s <seed> -l <log dir>

# example
python config/rmsc03_analysis.py -c rmsc03 -t ABM -d 20200603 -s 1234 -l rmsc03_two_hour_final
```

Proof of deterministic results
```bash
python abides/util/plotting/chart_fundamental.py -f "log/rmsc03_two_hour_new/impact/fundamental_ABM.bz2" -l impact1 \
-f "log/rmsc03_two_hour_final/impact/fundamental_ABM.bz2" -l impact2
```

## Agents



## Detection method

