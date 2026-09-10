# Medical Insurance Statistical Modeling Dashboard

### Lab 4 | Applied Statistical Modeling | DS602

[![Open Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://medical-insurance-statistical-modeling-dashboard.streamlit.app/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/apoorvayerpude/healthcare-dataset/data)

> An interactive statistical modeling dashboard for exploring medical insurance
> costs, testing group differences, fitting an OLS model, and evaluating its
> diagnostic assumptions.

## Live Resources

| Resource | Link |
|---|---|
| **Hosted dashboard** | [Open Insurance Analytics Dashboard](https://medical-insurance-statistical-modeling-dashboard.streamlit.app/) |
| **Dataset source** | [Healthcare Dataset on Kaggle](https://www.kaggle.com/datasets/apoorvayerpude/healthcare-dataset/data) |

## Project Overview
This project applies an end-to-end statistical workflow to a medical insurance
dataset containing demographic characteristics, health-related variables, and
insurance charges. The dashboard is designed for both exploratory analysis and
model-based inference, with interactive controls that update the analysis as
the selected sample changes.

```text
202618052_Lab_4/
|-- app.py                 # Interactive Streamlit dashboard
|-- analysis.py            # Reproducible command-line analysis
|-- requirements.txt       # Python dependencies
|-- data/
	|-- insurance.csv        # Local copy of the analysis dataset
```

## Dashboard Features

### 1. Data Exploration

- Interactive filters for age, BMI, smoking status, and region
- Summary metrics and descriptive statistics
- Histograms with KDE, bivariate scatter plots, and a correlation heatmap

### 2. Hypothesis Testing Lab

- Dynamic two-group comparison with selectable categorical and numerical
	variables
- Shapiro-Wilk normality testing and Levene's equal-variance testing
- Automatic selection of an independent t-test or Mann-Whitney U test
- Dynamic one-way ANOVA across selectable groups
- Clear decisions at significance level $\alpha = 0.05$

### 3. Prediction and Diagnostics

- Multiple linear regression using `statsmodels.api.OLS`
- Coefficients, p-values, 95% confidence intervals, R-squared, and adjusted
	R-squared
- Live charge prediction with 95% confidence and prediction intervals
- Residuals-versus-fitted plot and Q-Q plot
- Jarque-Bera residual normality test and VIF analysis

## Installation

Open PowerShell in this folder and run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `venv` already exists, activate it and install the dependencies with the
last two commands.

## Run Locally

Run all commands from the `202618052_Lab_4` directory.

### Launch the dashboard

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

Open `http://localhost:8501` in a browser.

The dashboard contains three tabs:

1. **Data Exploration**: Age, BMI, smoking-status, and region filters; summary
	 metrics; descriptive statistics; histograms; scatter plots; and a
	 correlation heatmap.
2. **Hypothesis Testing Lab**: Dynamic two-group comparisons and one-way ANOVA
	 with selectable categorical grouping variables and numerical outcomes.
3. **Prediction and Diagnostics**: OLS regression, coefficient inference,
	 live prediction intervals, residual plots, Jarque-Bera testing, and VIF.

### Run the reproducible analysis

For a console-based report and diagnostic figures:

```powershell
.\venv\Scripts\python.exe analysis.py
```

The script reads `data/insurance.csv` relative to the Lab 4 folder. Run it from
this folder so the data path resolves correctly.

## Statistical Methodology

### Exploratory analysis

The numerical variables are `age`, `bmi`, `children`, and `charges`.
For each variable, the project reports mean, median, standard deviation, Q1,
Q3, IQR, skewness, and kurtosis. The visual analysis includes KDE histograms,
age and BMI against charges, and a numerical correlation matrix.

### Hypothesis test 1

The default comparison is medical charges for smokers and non-smokers.
Shapiro-Wilk tests assess normality and Levene's test assesses equality of
variance. Because the groups are strongly non-normal, the dashboard selects a
two-sided Mann-Whitney U test. The interface also allows another categorical
factor, numerical metric, and pair of groups to be selected.

### Hypothesis test 2

The default ANOVA compares charges across the four regions. The dashboard also
allows the grouping variable and numerical outcome to be changed.

### Regression model

The OLS response is `charges`. Predictors are age, BMI, number of children,
sex, smoker status, and region. Categorical predictors are encoded with
one-hot encoding and one reference category is removed. The dashboard reports
coefficients, p-values, 95% confidence intervals, R-squared, and adjusted
R-squared.

The regression and residual diagnostics use the active filtered sample when at
least 30 observations remain. With fewer than 30 observations, the dashboard
uses the complete dataset to avoid unstable estimates.

## Main Findings

The following values come from the complete dataset of 1,338 observations.

- The dataset contains 7 variables, no missing values, and one duplicate row.
- Mean charges are approximately `$32,050.23` for smokers and `$8,434.27`
	for non-smokers.
- Shapiro-Wilk p-values are below 0.05 for both groups, so normality is not
	supported. Levene's test also rejects equal variance (`p < 0.001`).
- The Mann-Whitney U test is significant (`U = 284133.0`, `p < 0.001`). At
	alpha = 0.05, the null hypothesis is rejected: charges differ between the
	two smoking groups.
- Regional charge means range from approximately `$12,346.94` in the
	southwest to `$14,735.41` in the southeast. One-way ANOVA is significant
	(`F = 2.970`, `p = 0.0309`), so at least one regional mean differs.
- The multiple regression model has `R-squared = 0.7509` and adjusted
	`R-squared = 0.7494`.
- Holding the other predictors constant, age, BMI, children, and smoker status
	are statistically significant at the 0.05 level. Sex is not significant in
	this model (`p = 0.693`).
- The continuous-predictor VIF values are close to 1: age `1.014`, BMI `1.012`,
	and children `1.002`. This indicates little evidence of multicollinearity.
- The residual Jarque-Bera test is significant (`JB = 718.887`,
	`p < 0.001`), so the residuals do not appear normally distributed. The
	residual plots should therefore be interpreted as diagnostic evidence rather
	than proof that all OLS assumptions hold.

## Reproducibility Notes

- Use the project virtual environment for both installation and execution.
- Run commands from `202618052_Lab_4`, because both scripts use the relative
	path `data/insurance.csv`.
- The dashboard recalculates filtered summaries, hypothesis tests, regression,
	prediction intervals, and diagnostics when the selected data changes.

## Deployment

The dashboard is deployed with Streamlit Community Cloud. The application entry
point is `202618052_Lab_4/app.py`, and the repository-level `requirements.txt`
contains the packages required for deployment.

To deploy a copy of this project, select the repository, choose the `main`
branch, and set the main file path to:

```text
202618052_Lab_4/app.py
```

## Academic Context

This project was completed for **Lab 4: Applied Statistical Modeling** in the
course **Statistical Methods(DS602)**, M.Sc. Data Science,
Semester 1.
