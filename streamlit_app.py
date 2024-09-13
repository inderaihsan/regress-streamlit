import streamlit as st
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from scipy.stats import anderson
import io
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")
    if not modify:
        return df
    df = df.copy()
    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    modification_container = st.container()
    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]
    return df

# Caching data to prevent reloading
@st.cache_data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

@st.cache_resource
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]
    return vif_data[vif_data['feature'] != 'const']

def plot_regression_lines(X, y, data):
    st.write("Scatter plot between all dependent variables")
    X = [item for item in X if item in data.columns]
    num_features = len(X)
    num_columns = 4
    num_rows = int(np.ceil(num_features / num_columns))

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 5 * num_rows))
    axes = axes.flatten()

    for i, feature in enumerate(X):
        ax = axes[i]
        sns.regplot(x=data[feature], y=data[y], ax=ax)
        ax.set_title(f'Regression: {y} vs {feature}')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

def regression_analysis(X, y, data):
    if data.isna().values.any():
        st.warning("Warning! Detected missing values. or values with infinity! Attempting to remove them...")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=[*X, y], inplace=True)
        st.write("Missing and infinity values removal successful. Current number of rows:", len(data))
    vif_data = calculate_vif(sm.add_constant(data[X]))
    
    reg_X = sm.add_constant(data[X])
    regression = sm.OLS(data[y], reg_X).fit()

    st.subheader("Regression Summary:")
    st.write(regression.summary(alpha=0.1))

    st.subheader("VIF Data:")
    st.write(vif_data)

    y_pred = regression.predict(reg_X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=data[y])
    sns.lineplot(x=y_pred, y=y_pred, color='red')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Linear Regression')
    st.pyplot(plt.gcf())
    plt.clf()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=regression.resid)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    st.pyplot(plt.gcf())
    plt.clf()

    residuals = regression.resid
    ad_test = anderson(residuals, dist='norm')
    plt.figure(figsize=(10, 6))
    sns.kdeplot(residuals, fill=True)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('Residuals')
    plt.title('Residuals Distribution')
    st.pyplot(plt.gcf())
    plt.clf()

    plot_regression_lines(X, y, data)

    st.write("Anderson-Darling Test:")
    st.write(f"Test Statistic: {ad_test.statistic}")
    for critical_value, significance_level in zip(ad_test.critical_values, ad_test.significance_level):
        st.write(f"Critical Value at {significance_level}% significance level: {critical_value}")

    if ad_test.statistic > ad_test.critical_values[2]:  # 5% significance level
        st.write("Residuals are not normally distributed.")
    else:
        st.write("Residuals are normally distributed.")

    summary_str = regression.summary().as_text()
    summary_df = pd.DataFrame([x.split() for x in summary_str.splitlines()])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, index=False, header=False)

    st.download_button(
        label="Download Regression Summary as Excel",
        data=output.getvalue(),
        file_name="regression_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.title('Regression Analysis Tool')
st.text('A simple linear regression analysis tool.')
st.subheader('How to use this app?')
st.text('1. Upload your file by clicking the "Upload file" button.')
st.text('2. Choose the independent and dependent variables from the left sidebar.')
st.text('3. Get the results by clicking the "Start regression analysis" button.')

uploaded = st.file_uploader("Please upload your Excel file", type=['xlsx'])

if uploaded is not None:
    dataframe = load_data(uploaded)
    st.write("Great, here is the preview of your data.")
    st.write(dataframe.head(5))
    st.write(f"Number of rows: {len(dataframe)}")
    dataframe = filter_dataframe(dataframe)
    dataframe_model = dataframe.select_dtypes(include='number')
    st.sidebar.header("Regression Settings")
    independent_vars = st.sidebar.multiselect("Select independent variable(s) (X)", dataframe_model.columns)
    dependent_var = st.sidebar.selectbox("Select dependent variable (Y)", dataframe_model.columns)
    if st.button("Start regression analysis") and independent_vars and dependent_var:
        st.text('If you like this app, kindly click "share" or "star" on my GitHub.')
        regression_analysis(independent_vars, dependent_var, dataframe)
