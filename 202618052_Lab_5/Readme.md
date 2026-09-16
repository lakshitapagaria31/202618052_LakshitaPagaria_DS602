# DS602 Lab 5 - Computational Statistical Methods

## Overview

This repository contains the DS602 Lab 5 submission for the computational implementation and verification of fundamental statistical methods using Python. The accompanying notebook combines numerical simulation with visual comparison of empirical and theoretical distributions.

## Contents

- `202618052_Assignment05_DS602.ipynb`: Jupyter notebook containing the statistical experiments, Python implementation, visualizations, and observations.
- `Readme.md`: Project overview, methodology, requirements, and reproducibility instructions.

## Topics Covered

1. **Discrete and continuous uniform distributions**: Examines the behavior and representation of uniform random variables.
2. **Inverse-transform sampling**: Generates samples from probability distributions by applying the inverse of a cumulative distribution function to uniform random variables.
3. **Binomial convergence and the infinite limit**: Investigates how binomial behavior approaches its limiting distribution as the relevant parameters increase.
4. **Discrete convolution**: Demonstrates convolution computationally and studies its role in combining discrete random variables.

## Methodology

- Uses NumPy for random number generation, numerical operations, and vectorized computation.
- Uses Matplotlib to visualize empirical samples and compare simulation results with theoretical distributions.
- Initializes a fixed random-number-generator seed (`42`) to make simulation results reproducible.
- Prefers vectorized NumPy operations over explicit Python loops for concise and efficient computation.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Jupyter Notebook or JupyterLab

The required Python packages can be installed using:

```bash
pip install -r ../requirements.txt
```

## Reproducibility

1. Open `202618052_Assignment05_DS602.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
2. Ensure the required Python packages are installed.
3. Run the notebook cells sequentially from top to bottom.
4. Review the generated plots and observations for each statistical method.

## Implementation Notes

- The random-number generator is initialized with a fixed seed for consistent results across runs.
- The notebook configures a standard figure size and font size for readable plots.
- Numerical experiments are designed to connect theoretical statistical ideas with their computational behavior.

