# ============================================================
# LAB-4: APPLIED STATISTICAL MODELING
# Medical Insurance Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scipy.stats import (
    shapiro,
    levene,
    ttest_ind,
    mannwhitneyu,
    f_oneway,
    jarque_bera
)

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = Path(__file__).parent / "data" / "insurance.csv"

df = pd.read_csv(DATA_PATH)

NUMERICAL_COLUMNS = [
    "age",
    "bmi",
    "children",
    "charges"
]

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


sns.histplot(
    data=df,
    x="charges",
    kde=True
)

plt.title("Distribution of Medical Charges")
plt.xlabel("Medical Charges")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Histogram: Age
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    kde=True
)

plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Histogram: BMI
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="bmi",
    kde=True
)

plt.title("Distribution of BMI")
plt.xlabel("BMI")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Age vs Charges
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="charges",
    hue="smoker"
)

plt.title("Age vs Medical Charges")
plt.xlabel("Age")
plt.ylabel("Medical Charges")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# BMI vs Charges
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="bmi",
    y="charges",
    hue="smoker"
)

plt.title("BMI vs Medical Charges")
plt.xlabel("BMI")
plt.ylabel("Medical Charges")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Correlation Matrix
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

correlation = df[NUMERICAL_COLUMNS].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# 4. HYPOTHESIS TEST 1
# Smokers vs Non-Smokers
# ============================================================

print("\n" + "=" * 70)
print("HYPOTHESIS TEST 1")
print("Smokers vs Non-Smokers")
print("=" * 70)


# ------------------------------------------------------------
# Hypotheses
# H0: Mean charges are equal
# H1: Mean charges are different
# ------------------------------------------------------------

alpha = 0.05


smokers = df[
    df["smoker"] == "yes"
]["charges"]

non_smokers = df[
    df["smoker"] == "no"
]["charges"]


print("\nNumber of smokers:", len(smokers))
print("Number of non-smokers:", len(non_smokers))

print("\nMean smoker charges:")
print(smokers.mean())

print("\nMean non-smoker charges:")
print(non_smokers.mean())


# ------------------------------------------------------------
# Shapiro-Wilk Normality Test
# ------------------------------------------------------------

shapiro_smoker = shapiro(smokers)

shapiro_non_smoker = shapiro(
    non_smokers
)

print("\nShapiro-Wilk Test - Smokers")
print("Statistic:", shapiro_smoker.statistic)
print("p-value:", shapiro_smoker.pvalue)

print("\nShapiro-Wilk Test - Non-Smokers")
print(
    "Statistic:",
    shapiro_non_smoker.statistic
)

print(
    "p-value:",
    shapiro_non_smoker.pvalue
)


# ------------------------------------------------------------
# Levene's Test
# ------------------------------------------------------------

levene_result = levene(
    smokers,
    non_smokers
)

print("\nLevene's Test")
print("Statistic:", levene_result.statistic)
print("p-value:", levene_result.pvalue)


# ------------------------------------------------------------
# Select appropriate test
# ------------------------------------------------------------

if (
    shapiro_smoker.pvalue > alpha
    and
    shapiro_non_smoker.pvalue > alpha
):

    equal_variance = (
        levene_result.pvalue > alpha
    )

    test_result = ttest_ind(
        smokers,
        non_smokers,
        equal_var=equal_variance
    )

    test_name = "Independent Two-Sample t-test"

else:

    test_result = mannwhitneyu(
        smokers,
        non_smokers,
        alternative="two-sided"
    )

    test_name = "Mann-Whitney U test"


print("\nSelected Test:")
print(test_name)

print("\nTest Statistic:")
print(test_result.statistic)

print("\np-value:")
print(test_result.pvalue)


# ------------------------------------------------------------
# Final decision
# ------------------------------------------------------------

if test_result.pvalue < alpha:

    print(
        "\nDecision: Reject H0"
    )

    print(
        "Conclusion: There is statistically "
        "significant evidence of a difference "
        "between smoker and non-smoker charges."
    )

else:

    print(
        "\nDecision: Fail to Reject H0"
    )

    print(
        "Conclusion: There is insufficient "
        "evidence of a statistically significant "
        "difference between smoker and non-smoker charges."
    )


# ============================================================
# 5. HYPOTHESIS TEST 2
# ONE-WAY ANOVA: Region vs Charges
# ============================================================

print("\n" + "=" * 70)
print("HYPOTHESIS TEST 2")
print("One-Way ANOVA: Charges across Regions")
print("=" * 70)


# ------------------------------------------------------------
# Create groups
# ------------------------------------------------------------

region_groups = []

for region in sorted(
    df["region"].unique()
):

    group = df[
        df["region"] == region
    ]["charges"]

    region_groups.append(group)

    print(
        f"\n{region}:"
    )

    print(
        "Number of observations:",
        len(group)
    )

    print(
        "Mean:",
        group.mean()
    )


# ------------------------------------------------------------
# ANOVA
# ------------------------------------------------------------

anova_result = f_oneway(
    *region_groups
)

print("\nANOVA F-statistic:")
print(anova_result.statistic)

print("\nANOVA p-value:")
print(anova_result.pvalue)


# ------------------------------------------------------------
# ANOVA Decision
# ------------------------------------------------------------

if anova_result.pvalue < alpha:

    print(
        "\nDecision: Reject H0"
    )

    print(
        "Conclusion: At least one region "
        "has a significantly different "
        "mean medical charge."
    )

else:

    print(
        "\nDecision: Fail to Reject H0"
    )

    print(
        "Conclusion: There is insufficient "
        "evidence that the regional mean "
        "charges differ."
    )


# ============================================================
# 6. MULTIPLE LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 70)


# ------------------------------------------------------------
# Predictor variables
# ------------------------------------------------------------

predictor_cols = [
    "age",
    "bmi",
    "children",
    "sex",
    "smoker",
    "region"
]


# ------------------------------------------------------------
# Convert categorical variables to dummy variables
# ------------------------------------------------------------

X = pd.get_dummies(
    df[predictor_cols],
    drop_first=True,
    dtype=float
)


# Target variable

y = df["charges"]


# ------------------------------------------------------------
# Add intercept
# ------------------------------------------------------------

X = sm.add_constant(X)


# ------------------------------------------------------------
# Fit OLS model
# ------------------------------------------------------------

model = sm.OLS(
    y,
    X
).fit()


# ------------------------------------------------------------
# Model summary
# ------------------------------------------------------------

print(model.summary())


# ============================================================
# 7. MODEL PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)


print("\nCoefficients:")
print(model.params)


print("\nP-values:")
print(model.pvalues)


print("\n95% Confidence Intervals:")
print(model.conf_int())


print("\nR-squared:")
print(model.rsquared)


print("\nAdjusted R-squared:")
print(model.rsquared_adj)


# ============================================================
# 8. PREDICTIONS AND RESIDUALS
# ============================================================

predictions = model.fittedvalues

residuals = model.resid


print("\n" + "=" * 70)
print("PREDICTIONS AND RESIDUALS")
print("=" * 70)


print("\nFirst five predictions:")
print(predictions.head())


print("\nFirst five residuals:")
print(residuals.head())


# ============================================================
# 9. RESIDUALS VS FITTED
# Linearity + Homoscedasticity
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=predictions,
    y=residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title(
    "Residuals vs Fitted Values"
)

plt.xlabel(
    "Fitted Values"
)

plt.ylabel(
    "Residuals"
)

plt.tight_layout()
plt.show()


# ============================================================
# 10. Q-Q PLOT
# ============================================================

print("\n" + "=" * 70)
print("Q-Q PLOT OF RESIDUALS")
print("=" * 70)


fig, ax = plt.subplots(figsize=(8, 5))

sm.qqplot(
    residuals,
    line="45",
    fit=True,
    ax=ax
)

ax.set_title("Q-Q Plot of Regression Residuals")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Ordered Residuals")
ax.grid(alpha=0.25)

fig.tight_layout()
plt.show()
plt.close(fig)
# ============================================================
# 11. JARQUE-BERA TEST
# ============================================================

print("\n" + "=" * 70)
print("JARQUE-BERA TEST")
print("=" * 70)


jb_result = jarque_bera(
    residuals
)


print(
    "Jarque-Bera statistic:",
    jb_result.statistic
)

print(
    "Jarque-Bera p-value:",
    jb_result.pvalue
)


if jb_result.pvalue < alpha:

    print(
        "Decision: Reject normality H0."
    )

    print(
        "Residuals show evidence of "
        "departure from normality."
    )

else:

    print(
        "Decision: Fail to Reject normality H0."
    )

    print(
        "There is insufficient evidence "
        "against residual normality."
    )


# ============================================================
# 12. VIF
# Multicollinearity
# ============================================================

print("\n" + "=" * 70)
print("VARIANCE INFLATION FACTOR")
print("=" * 70)


# The lab specifically asks for VIF
# for continuous predictors.

X_vif = df[
    [
        "age",
        "bmi",
        "children"
    ]
].copy()


X_vif = sm.add_constant(
    X_vif
)


vif_results = pd.DataFrame()

vif_results["Variable"] = (
    X_vif.columns
)

vif_results["VIF"] = [
    variance_inflation_factor(
        X_vif.values,
        i
    )
    for i in range(
        X_vif.shape[1]
    )
]


print(
    vif_results.to_string(
        index=False
    )
)


# ============================================================
# 13. OPTIONAL INTERACTION MODEL
# BMI × SMOKER
# ============================================================

print("\n" + "=" * 70)
print("OPTIONAL INTERACTION MODEL")
print("BMI × Smoker")
print("=" * 70)


df["smoker_num"] = (
    df["smoker"] == "yes"
).astype(int)


df["bmi_smoker"] = (
    df["bmi"] *
    df["smoker_num"]
)


interaction_X = df[
    [
        "age",
        "bmi",
        "children",
        "smoker_num",
        "bmi_smoker"
    ]
]


interaction_X = sm.add_constant(
    interaction_X
)


interaction_model = sm.OLS(
    df["charges"],
    interaction_X
).fit()


print(
    interaction_model.summary()
)


print("\nInteraction coefficient:")
print(
    interaction_model.params[
        "bmi_smoker"
    ]
)


print("\nInteraction p-value:")
print(
    interaction_model.pvalues[
        "bmi_smoker"
    ]
)


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)