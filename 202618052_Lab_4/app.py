# ============================================================
# LAB-4: APPLIED STATISTICAL MODELING
# INTERACTIVE STREAMLIT DASHBOARD
# Medical Insurance Dataset
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Insurance Statistical Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 3. TITLE
# ============================================================

st.title(
    "📊 Medical Insurance Statistical Modeling Dashboard"
)

st.write(
    """
    This dashboard performs complete statistical analysis
    of medical insurance charges including:

    • Exploratory Data Analysis
    • Hypothesis Testing
    • Multiple Linear Regression
    • Residual Diagnostics
    • Multicollinearity Analysis
    • Live Prediction
    """
)


# ============================================================
# 4. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        Path(__file__).parent / "data" / "insurance.csv"
    )

    return data


df = load_data()

NUMERIC_COLUMNS = [
    "age",
    "bmi",
    "children",
    "charges"
]

CATEGORICAL_COLUMNS = [
    "sex",
    "smoker",
    "region"
]


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.title(
    "Dashboard Controls"
)

st.sidebar.write(
    "Use the filters and tabs to explore the analysis."
)


# ============================================================
# 6. CREATE THREE TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Data Exploration",
        "🧪 Hypothesis Testing",
        "🔮 Prediction & Diagnostics"
    ]
)


# ============================================================
# TAB 1
# DATA EXPLORATION
# ============================================================

with tab1:

    st.header(
        "📊 Data Exploration"
    )

    st.write(
        """
        Explore the medical insurance dataset
        using interactive filters and visualizations.
        """
    )


    # ========================================================
    # SIDEBAR FILTERS
    # ========================================================

    st.sidebar.header(
        "Data Filters"
    )


    # --------------------------------------------------------
    # AGE FILTER
    # --------------------------------------------------------

    min_age = int(
        df["age"].min()
    )

    max_age = int(
        df["age"].max()
    )


    age_range = st.sidebar.slider(
        "Age Range",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )


    # --------------------------------------------------------
    # BMI FILTER
    # --------------------------------------------------------

    min_bmi = float(
        df["bmi"].min()
    )

    max_bmi = float(
        df["bmi"].max()
    )


    bmi_range = st.sidebar.slider(
        "BMI Range",
        min_value=float(
            round(min_bmi, 1)
        ),
        max_value=float(
            round(max_bmi, 1)
        ),
        value=(
            float(round(min_bmi, 1)),
            float(round(max_bmi, 1))
        )
    )


    # --------------------------------------------------------
    # SMOKER FILTER
    # --------------------------------------------------------

    smoker_options = sorted(
        df["smoker"].unique()
    )


    selected_smokers = st.sidebar.multiselect(
        "Smoking Status",
        options=smoker_options,
        default=smoker_options
    )


    # --------------------------------------------------------
    # REGION FILTER
    # --------------------------------------------------------

    region_options = sorted(
        df["region"].unique()
    )


    selected_regions = st.sidebar.multiselect(
        "Region",
        options=region_options,
        default=region_options
    )


    # ========================================================
    # FILTER DATA
    # ========================================================

    filtered_df = df[
        (df["age"] >= age_range[0])
        &
        (df["age"] <= age_range[1])
        &
        (df["bmi"] >= bmi_range[0])
        &
        (df["bmi"] <= bmi_range[1])
        &
        (df["smoker"].isin(selected_smokers))
        &
        (df["region"].isin(selected_regions))
    ]


    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    st.subheader(
        "Dataset Summary"
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Observations",
        len(filtered_df)
    )


    if len(filtered_df) > 0:

        col2.metric(
            "Mean Charges",
            f"${filtered_df['charges'].mean():,.2f}"
        )


        col3.metric(
            "Median Charges",
            f"${filtered_df['charges'].median():,.2f}"
        )


        col4.metric(
            "Mean BMI",
            f"{filtered_df['bmi'].mean():.2f}"
        )

    else:

        col2.metric(
            "Mean Charges",
            "N/A"
        )

        col3.metric(
            "Median Charges",
            "N/A"
        )

        col4.metric(
            "Mean BMI",
            "N/A"
        )


    # ========================================================
    # DISPLAY FILTERED DATA
    # ========================================================

    st.subheader(
        "Filtered Dataset"
    )


    st.dataframe(
        filtered_df,
        width="stretch"
    )


    if len(filtered_df) == 0:

        st.warning(
            "No observations match the selected filters."
        )

    else:

        # ====================================================
        # DESCRIPTIVE STATISTICS
        # ====================================================

        st.subheader(
            "Descriptive Statistics"
        )


        numerical_cols = [
            "age",
            "bmi",
            "children",
            "charges"
        ]


        descriptive = (
            filtered_df[
                numerical_cols
            ]
            .describe()
            .T
        )


        # IQR

        descriptive["IQR"] = (
            filtered_df[
                numerical_cols
            ].quantile(0.75)
            -
            filtered_df[
                numerical_cols
            ].quantile(0.25)
        )


        # Skewness

        descriptive["Skewness"] = (
            filtered_df[
                numerical_cols
            ].skew()
        )


        # Kurtosis

        descriptive["Kurtosis"] = (
            filtered_df[
                numerical_cols
            ].kurt()
        )


        st.dataframe(
            descriptive,
            width="stretch"
        )


        # ====================================================
        # CHARGES HISTOGRAM
        # ====================================================

        st.subheader(
            "Distribution of Medical Charges"
        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        sns.histplot(
            data=filtered_df,
            x="charges",
            kde=True,
            ax=ax
        )


        ax.set_title(
            "Distribution of Medical Charges"
        )

        ax.set_xlabel(
            "Medical Charges"
        )

        ax.set_ylabel(
            "Frequency"
        )


        st.pyplot(fig)

        plt.close(fig)


        # ====================================================
        # AGE VS CHARGES
        # ====================================================

        st.subheader(
            "Age vs Medical Charges"
        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        sns.scatterplot(
            data=filtered_df,
            x="age",
            y="charges",
            hue="smoker",
            ax=ax
        )


        ax.set_title(
            "Age vs Medical Charges"
        )


        st.pyplot(fig)

        plt.close(fig)


        # ====================================================
        # BMI VS CHARGES
        # ====================================================

        st.subheader(
            "BMI vs Medical Charges"
        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        sns.scatterplot(
            data=filtered_df,
            x="bmi",
            y="charges",
            hue="smoker",
            ax=ax
        )


        ax.set_title(
            "BMI vs Medical Charges"
        )


        st.pyplot(fig)

        plt.close(fig)


        # ====================================================
        # CORRELATION MATRIX
        # ====================================================

        st.subheader(
            "Correlation Matrix"
        )


        correlation_cols = [
            "age",
            "bmi",
            "children",
            "charges"
        ]


        correlation = (
            filtered_df[
                correlation_cols
            ].corr()
        )


        fig, ax = plt.subplots(
            figsize=(8, 6)
        )


        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )


        ax.set_title(
            "Correlation Matrix"
        )


        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# Use the active exploration sample for modeling when it is large enough
# to support stable regression and diagnostic estimates.
modeling_df = (
    filtered_df.copy()
    if len(filtered_df) >= 30
    else df.copy()
)


# TAB 2
# HYPOTHESIS TESTING
# ============================================================

with tab2:

    st.header(
        "🧪 Hypothesis Testing Lab"
    )


    st.info(
        "Significance level: α = 0.05"
    )


    # ========================================================
    # HYPOTHESIS TEST 1
    # ========================================================

    st.subheader(
        "Test 1: Compare Two Groups"
    )

    group_column = st.selectbox(
        "Grouping variable",
        CATEGORICAL_COLUMNS,
        index=CATEGORICAL_COLUMNS.index("smoker")
    )

    metric_column = st.selectbox(
        "Numerical variable",
        NUMERIC_COLUMNS,
        index=NUMERIC_COLUMNS.index("charges")
    )

    group_values = sorted(
        df[group_column].dropna().unique().tolist()
    )

    if len(group_values) < 2:
        st.warning("The selected grouping variable does not contain two groups.")
        st.stop()

    selected_groups = st.multiselect(
        "Select exactly two groups",
        group_values,
        default=group_values[:2]
    )

    if len(selected_groups) != 2:
        st.warning("Select exactly two groups to run the comparison.")
        st.stop()

    first_group, second_group = selected_groups


    st.markdown(
        """
        **H₀:** There is no significant difference
        in the selected numerical variable between
        the two selected groups.

        **H₁:** There is a significant difference
        in the selected numerical variable between
        the two selected groups.
        """
    )


    # --------------------------------------------------------
    # CREATE GROUPS
    # --------------------------------------------------------

    first_sample = df.loc[
        df[group_column] == first_group,
        metric_column
    ].dropna()

    second_sample = df.loc[
        df[group_column] == second_group,
        metric_column
    ].dropna()


    # --------------------------------------------------------
    # GROUP MEANS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    col1.metric(
        f"Mean {metric_column} - {first_group}",
        f"{first_sample.mean():,.2f}"
    )


    col2.metric(
        f"Mean {metric_column} - {second_group}",
        f"{second_sample.mean():,.2f}"
    )


    # ========================================================
    # SHAPIRO-WILK TEST
    # ========================================================

    st.markdown(
        "### 1. Shapiro-Wilk Normality Test"
    )


    shapiro_first = stats.shapiro(
        first_sample
    )


    shapiro_second = stats.shapiro(
        second_sample
    )


    shapiro_results = pd.DataFrame(
        {
            "Group": [
                first_group,
                second_group
            ],
            "Statistic": [
                shapiro_first.statistic,
                shapiro_second.statistic
            ],
            "p-value": [
                shapiro_first.pvalue,
                shapiro_second.pvalue
            ]
        }
    )


    st.dataframe(
        shapiro_results,
        width="stretch"
    )


    # ========================================================
    # LEVENE'S TEST
    # ========================================================

    st.markdown(
        "### 2. Levene's Test"
    )


    levene_result = stats.levene(
        first_sample,
        second_sample
    )


    st.write(
        "Levene statistic:",
        levene_result.statistic
    )


    st.write(
        "Levene p-value:",
        levene_result.pvalue
    )


    if levene_result.pvalue > 0.05:

        st.success(
            "p > 0.05: Equal variance assumption is reasonable."
        )

    else:

        st.warning(
            "p ≤ 0.05: The variances are significantly different."
        )


    # ========================================================
    # AUTOMATIC TEST SELECTION
    # ========================================================

    st.markdown(
        "### 3. Statistical Test"
    )


    if (
        shapiro_first.pvalue > 0.05
        and
        shapiro_second.pvalue > 0.05
    ):

        # Both groups are approximately normal

        equal_variance = (
            levene_result.pvalue > 0.05
        )


        test_result = stats.ttest_ind(
            first_sample,
            second_sample,
            equal_var=equal_variance
        )


        test_name = (
            "Independent Two-Sample t-test"
        )


    else:

        # At least one group is not approximately normal

        test_result = stats.mannwhitneyu(
            first_sample,
            second_sample,
            alternative="two-sided"
        )


        test_name = (
            "Mann-Whitney U test"
        )


    st.info(
        f"Selected Test: {test_name}"
    )


    st.write(
        "Test statistic:",
        test_result.statistic
    )


    st.write(
        "p-value:",
        test_result.pvalue
    )


    # ========================================================
    # FINAL DECISION
    # ========================================================

    st.markdown(
        "### 4. Final Decision"
    )


    if test_result.pvalue < 0.05:

        st.error(
            """
            **Reject H₀**

            There is statistically significant evidence
            of a difference between the selected groups.
            """
        )

    else:

        st.success(
            """
            **Fail to Reject H₀**

            There is insufficient evidence of a
            statistically significant difference between
            the selected groups.
            """
        )


    # ========================================================
    # HYPOTHESIS TEST 2
    # ONE-WAY ANOVA
    # ========================================================

    st.divider()


    st.subheader(
        "Test 2: One-Way ANOVA"
    )

    anova_group_column = st.selectbox(
        "Grouping variable for ANOVA",
        CATEGORICAL_COLUMNS,
        index=CATEGORICAL_COLUMNS.index("region")
    )

    anova_metric_column = st.selectbox(
        "Numerical variable for ANOVA",
        NUMERIC_COLUMNS,
        index=NUMERIC_COLUMNS.index("charges")
    )


    st.markdown(
        """
        **H₀:** All group means of the selected
        numerical variable are equal.

        **H₁:** At least one group mean is different.
        """
    )


    # --------------------------------------------------------
    # CREATE REGION GROUPS
    # --------------------------------------------------------

    region_groups = []


    region_summary = []


    for group in sorted(
        df[anova_group_column].dropna().unique()
    ):

        values = df[
            df[anova_group_column] == group
        ][anova_metric_column].dropna()


        region_groups.append(
            values
        )


        region_summary.append(
            {
                "Group": group,
                "Count": len(values),
                f"Mean {anova_metric_column}": values.mean(),
                f"Median {anova_metric_column}": values.median()
            }
        )


    region_summary_df = pd.DataFrame(
        region_summary
    )


    st.dataframe(
        region_summary_df,
        width="stretch"
    )


    # --------------------------------------------------------
    # PERFORM ANOVA
    # --------------------------------------------------------

    anova_result = stats.f_oneway(
        *region_groups
    )


    st.write(
        "F-statistic:",
        anova_result.statistic
    )


    st.write(
        "p-value:",
        anova_result.pvalue
    )


    # --------------------------------------------------------
    # ANOVA DECISION
    # --------------------------------------------------------

    if anova_result.pvalue < 0.05:

        st.error(
            """
            **Reject H₀**

            At least one region has a significantly
            different mean medical charge.
            """
        )

    else:

        st.success(
            """
            **Fail to Reject H₀**

            There is insufficient evidence that the
            regional mean medical charges differ.
            """
        )


# ============================================================
# TAB 3
# PREDICTION & DIAGNOSTICS
# ============================================================

with tab3:

    st.header(
        "🔮 Live Prediction & Diagnostics"
    )


    st.write(
        """
        This section fits a multiple linear regression
        model using OLS and allows live prediction of
        medical insurance charges.
        """
    )


    # ========================================================
    # BUILD OLS MODEL
    # ========================================================

    predictor_cols = [
        "age",
        "bmi",
        "children",
        "sex",
        "smoker",
        "region"
    ]


    # --------------------------------------------------------
    # ENCODE CATEGORICAL VARIABLES
    # --------------------------------------------------------

    model_data = pd.get_dummies(
        modeling_df[predictor_cols],
        drop_first=True,
        dtype=float
    )


    # --------------------------------------------------------
    # TARGET VARIABLE
    # --------------------------------------------------------

    y = modeling_df["charges"]


    # --------------------------------------------------------
    # ADD INTERCEPT
    # --------------------------------------------------------

    X = sm.add_constant(
        model_data
    )


    # --------------------------------------------------------
    # FIT OLS
    # --------------------------------------------------------

    model = sm.OLS(
        y,
        X
    ).fit()


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader(
        "Regression Model Performance"
    )


    col1, col2 = st.columns(2)


    col1.metric(
        "R²",
        f"{model.rsquared:.4f}"
    )


    col2.metric(
        "Adjusted R²",
        f"{model.rsquared_adj:.4f}"
    )


    # ========================================================
    # MODEL COEFFICIENTS
    # ========================================================

    st.subheader(
        "Regression Coefficients"
    )


    confidence_intervals = (
        model.conf_int()
    )


    coefficient_table = pd.DataFrame(
        {
            "Coefficient": model.params,
            "P-value": model.pvalues,
            "95% CI Lower": confidence_intervals[0],
            "95% CI Upper": confidence_intervals[1]
        }
    )


    st.dataframe(
        coefficient_table,
        width="stretch"
    )


    # ========================================================
    # USER INPUTS
    # ========================================================

    st.subheader(
        "Enter Person Details"
    )


    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        age = st.slider(
            "Age",
            min_value=int(
                modeling_df["age"].min()
            ),
            max_value=int(
                modeling_df["age"].max()
            ),
            value=30
        )


        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=70.0,
            value=25.0,
            step=0.1
        )


    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )


        sex = st.selectbox(
            "Sex",
            sorted(
                modeling_df["sex"].unique()
            )
        )


    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        smoker = st.selectbox(
            "Smoker",
            sorted(
                modeling_df["smoker"].unique()
            )
        )


        region = st.selectbox(
            "Region",
            sorted(
                modeling_df["region"].unique()
            )
        )


    # ========================================================
    # CREATE NEW PERSON DATA
    # ========================================================

    input_data = pd.DataFrame(
        {
            "age": [age],
            "bmi": [bmi],
            "children": [children],
            "sex": [sex],
            "smoker": [smoker],
            "region": [region]
        }
    )


    # --------------------------------------------------------
    # ENCODE INPUT
    # --------------------------------------------------------

    input_encoded = pd.get_dummies(
        input_data,
        drop_first=True,
        dtype=float
    )


    # --------------------------------------------------------
    # ENSURE SAME COLUMNS AS TRAINING DATA
    # --------------------------------------------------------

    input_encoded = input_encoded.reindex(
        columns=model_data.columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # ADD INTERCEPT
    # --------------------------------------------------------

    input_encoded = sm.add_constant(
        input_encoded,
        has_constant="add"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if st.button(
        "🔮 Predict Medical Charges",
        type="primary"
    ):

        prediction = model.get_prediction(
            input_encoded
        )


        # ----------------------------------------------------
        # 95% INTERVALS
        # ----------------------------------------------------

        prediction_summary = (
            prediction.summary_frame(
                alpha=0.05
            )
        )


        predicted_value = (
            prediction_summary[
                "mean"
            ].iloc[0]
        )


        confidence_lower = (
            prediction_summary[
                "mean_ci_lower"
            ].iloc[0]
        )


        confidence_upper = (
            prediction_summary[
                "mean_ci_upper"
            ].iloc[0]
        )


        prediction_lower = (
            prediction_summary[
                "obs_ci_lower"
            ].iloc[0]
        )


        prediction_upper = (
            prediction_summary[
                "obs_ci_upper"
            ].iloc[0]
        )


        # ----------------------------------------------------
        # DISPLAY PREDICTION
        # ----------------------------------------------------

        st.subheader(
            "Prediction Result"
        )


        st.success(
            f"Predicted Medical Charges: "
            f"${predicted_value:,.2f}"
        )


        col1, col2 = st.columns(2)


        col1.metric(
            "95% Confidence Interval",
            f"${confidence_lower:,.2f} – "
            f"${confidence_upper:,.2f}"
        )


        col2.metric(
            "95% Prediction Interval",
            f"${prediction_lower:,.2f} – "
            f"${prediction_upper:,.2f}"
        )


    # ========================================================
    # RESIDUAL DIAGNOSTICS
    # ========================================================

    st.divider()


    st.header(
        "📈 Residual Diagnostics"
    )


    # --------------------------------------------------------
    # GET FITTED VALUES
    # AND RESIDUALS
    # --------------------------------------------------------

    fitted_values = model.fittedvalues

    residuals = model.resid


    # ========================================================
    # DIAGNOSTIC 1
    # RESIDUALS VS FITTED
    # ========================================================

    st.subheader(
        "1. Residuals vs Fitted Values"
    )


    st.write(
        """
        This plot is used to investigate:
        
        • Linearity
        • Homoscedasticity
        
        Ideally, residuals should be randomly scattered
        around zero with approximately constant spread.
        """
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    sns.scatterplot(
        x=fitted_values,
        y=residuals,
        ax=ax
    )


    ax.axhline(
        y=0,
        linestyle="--"
    )


    ax.set_title(
        "Residuals vs Fitted Values"
    )


    ax.set_xlabel(
        "Fitted Values"
    )


    ax.set_ylabel(
        "Residuals"
    )


    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # DIAGNOSTIC 2
    # Q-Q PLOT
    # ========================================================

    st.subheader(
        "2. Q-Q Plot of Residuals"
    )


    st.write(
        """
        The Q-Q plot is generated AFTER fitting the OLS model.

        It checks whether the regression residuals are
        approximately normally distributed.

        If the points approximately follow the diagonal
        reference line, residual normality is more plausible.
        """
    )


    # --------------------------------------------------------
    # CREATE Q-Q PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    sm.qqplot(
        residuals,
        line="45",
        fit=True,
        ax=ax
    )


    ax.set_title(
        "Q-Q Plot of Residuals"
    )


    ax.set_xlabel(
        "Theoretical Quantiles"
    )


    ax.set_ylabel(
        "Sample Quantiles"
    )

    ax.grid(alpha=0.25)
    fig.tight_layout()


    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # DIAGNOSTIC 3
    # JARQUE-BERA TEST
    # ========================================================

    st.subheader(
        "3. Jarque-Bera Normality Test"
    )


    jb_result = stats.jarque_bera(
        residuals
    )


    col1, col2 = st.columns(2)


    col1.metric(
        "Jarque-Bera Statistic",
        f"{jb_result.statistic:.4f}"
    )


    col2.metric(
        "p-value",
        f"{jb_result.pvalue:.6f}"
    )


    if jb_result.pvalue < 0.05:

        st.warning(
            """
            **Reject H₀**

            The residuals show statistically significant
            evidence of departure from normality.
            """
        )

    else:

        st.success(
            """
            **Fail to Reject H₀**

            There is insufficient statistical evidence
            against residual normality.
            """
        )


    # ========================================================
    # DIAGNOSTIC 4
    # VIF
    # ========================================================

    st.subheader(
        "4. Variance Inflation Factor (VIF)"
    )


    st.write(
        """
        VIF is used to investigate multicollinearity
        among the continuous predictors.
        """
    )


    vif_input = modeling_df[
        [
            "age",
            "bmi",
            "children"
        ]
    ].copy()


    vif_input = sm.add_constant(
        vif_input
    )


    vif_table = pd.DataFrame()


    vif_table["Variable"] = (
        vif_input.columns
    )


    vif_table["VIF"] = [
        variance_inflation_factor(
            vif_input.values,
            i
        )
        for i in range(
            vif_input.shape[1]
        )
    ]


    st.dataframe(
        vif_table,
        width="stretch"
    )


    # ========================================================
    # VIF INTERPRETATION
    # ========================================================

    st.write(
        """
        General rule of thumb:

        • VIF ≈ 1 → little multicollinearity
        • VIF between 1 and 5 → usually acceptable
        • VIF > 5 → potentially concerning
        • VIF > 10 → serious multicollinearity concern

        These are practical guidelines rather than strict rules.
        """
    )



