# DS602 Lab 1 - Statistical Analysis and Visualization

## Overview
This repository contains the DS602 lab submission for exploratory data analysis and statistical visualization. The project uses the Palmer Penguins and Dow Jones datasets to illustrate key concepts in descriptive statistics, distribution analysis, correlation, and trend evaluation.

## Contents
- `202618052_Assignment01_DS602.ipynb`: Jupyter notebook presenting the analysis workflow, supporting code, plots, and detailed observations.
- `Readme.md`: Project description, methodology, and summarized findings.

## Data Sources
- **Palmer Penguins**: A biological dataset used for species-level biometric analysis, distribution comparison, and group-based statistical evaluation.
- **Dow Jones Time Series**: A financial dataset used to examine long-term index behavior, decade comparisons, and regression-based trend analysis.

## Methodology
### Palmer Penguins Analysis
- Evaluated the effect of an extreme outlier on the mean and median of `body_mass_g`.
- Compared flipper length distributions across penguin species using histograms.
- Computed variance, skewness, kurtosis, and Pearson skewness coefficients for species-specific body mass.
- Generated five-number summaries, interquartile ranges, and grouped boxplots for body mass by species.
- Analyzed the linear relationship between `flipper_length_mm` and `body_mass_g` through regression.
- Summarized total body mass by island and visualized island-level biomass contributions.
- Examined species-specific patterns in bill length and bill depth using multivariable scatter plots.

### Dow Jones Time-Series Analysis
- Calculated five-number summaries and IQR of index prices.
- Engineered a decade-based feature to compare price distributions across decades.
- Fitted a linear regression model to the Dow Jones time series and evaluated trend strength using Pearson correlation and R-squared.

## Key Findings
- The mean is sensitive to outliers, while the median remains a stable measure of central tendency.
- Gentoo penguins display the highest variability in body mass and distinct biometric characteristics.
- Flipper length and body mass show a strong positive linear relationship.
- Biomass distribution across islands reflects species composition and population structure.
- Dow Jones prices exhibit increasing dispersion and upward movement over successive decades, with a linear model capturing the broader growth trend.

## Requirements
- Python 3.x
- pandas
- seaborn
- matplotlib
- scipy

## Reproducibility
1. Open `202618052_Assignment01_DS602.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the notebook cells sequentially.
3. Review the observations included after each analysis task.

## Notes
- The notebook loads datasets directly through `seaborn` to ensure consistent reproduction.
- The analysis combines code, visualization, and interpretation to support academic evaluation.
