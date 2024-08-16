import streamlit as st
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from scipy.stats import anderson
import io
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Define the regression analysis function

def plot_regression_lines(X, y, data):
    # print(X) 
    # print(y)
    st.write("Scatter plot between all dependent variable")
    X = [items for items in X if items in dataframe.columns]
    num_features = len(X)
    num_columns = 4
    num_rows = int(np.ceil(num_features / num_columns))

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 5 * num_rows))
    axes = axes.flatten()  # Flatten to make indexing easier

    for i, feature in enumerate(X):
        ax = axes[i]
        sns.regplot(x=data[feature], y=data[y], ax=ax)
        ax.set_title(f'Regression: {y} vs {feature}')

    # Remove any empty subplots if features are less than rows*columns
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)
    
def regression_analysis(X, y, data):
    columns = X.copy()
    if data.isna().values.any():
        st.subheader("Warning! detected a missing value!...")
        st.text("attempting to remove missing value from dataframe...")
        st.write(data[X].isnull().sum())
        # Dropping missing values from dataframe
        for column in X:
            if data[column].isna().any():
                data.dropna(subset=[column], inplace=True)
                # st.write(f"Removed missing values from column: {column}")
            data.dropna(subset = [y], inplace = True)
                
        st.write("missing value removal success... current number of rows : {}".format(len(dataframe)))

    vif__ = data[columns].copy()
    vif__ = sm.add_constant(vif__)
    vif_data = pd.DataFrame()
    vif_data["feature"] = vif__.columns
    vif_data["VIF"] = [variance_inflation_factor(vif__.values, i) for i in range(len(vif__.columns))]
    vif_data = vif_data[vif_data['feature']!='const']

    columns.append(y)
    data_copy = data[columns]
    data_copy.dropna(inplace=True)

    if y in X:
        X = [x for x in X if x != y]

    dependent = X 
    independent = y
    X = data_copy[X]
    y = data_copy[y]     
    
    

    reg_X = sm.add_constant(X)
    regression = sm.OLS(y, reg_X).fit()
    st.subheader("Regression Summary:")
    st.write(regression.summary(alpha=0.1))
    
    st.subheader("VIF Data:")
    st.write(vif_data)

    y_pred = regression.predict(reg_X)
    

    # Plot the real data points and the regression line
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=y)
    sns.lineplot(x=y_pred, y=y_pred, color='red')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Linear Regression')
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the figure after rendering

    # Residuals vs Fitted plot (RVF plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=regression.resid)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    st.pyplot(plt.gcf())
    plt.clf()

    # Normality of residuals
    residuals = regression.resid
    ad_test = anderson(residuals, dist='norm')
    plt.figure(figsize=(10, 6))
    sns.kdeplot(residuals, fill=True)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('Residuals')
    plt.title('Residuals Distribution')
    st.pyplot(plt.gcf())
    plt.clf()
    
    #variable relationship 
    
    plot_regression_lines(dependent, independent, dataframe)

    st.write("Anderson-Darling Test:")
    st.write(f"Test Statistic: {ad_test.statistic}")
    for critical_value, significance_level in zip(ad_test.critical_values, ad_test.significance_level):
        st.write(f"Critical Value at {significance_level}% significance level: {critical_value}")

    if ad_test.statistic > ad_test.critical_values[2]:  # 5% significance level
        st.write("Residuals are not normally distributed.")
    else:
        st.write("Residuals are normally distributed.")


    # Prepare the summary for download
    summary_str = regression.summary().as_text()
    summary_df = pd.DataFrame([x.split() for x in summary_str.splitlines()])
    
    # Save to Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, index=False, header=False)
        # writer.save()

    st.download_button(
        label="Download Regression Summary as Excel",
        data=output.getvalue(),
        file_name="regression_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Streamlit app
st.title('Regression Analysis Tool')
st.text('A simple linear regression analysis tools.') 
st.text('this is a prototype and still on development') 
st.subheader('How to use this app?')  
st.text('simply upload your file by clicking the form upload file')  
st.text('choose the independent and dependent variable on the left sidebar')  
st.text('get the result by clicking the submit button')  

uploaded = st.file_uploader(label="Please Upload your Excel file", type=['xlsx'])


if uploaded is not None:
    dataframe = pd.read_excel(uploaded)
    st.write("Great, here is the preview of your data.")
    st.write(dataframe.head(5))
    dataframe_model = dataframe.select_dtypes(include='number')

    st.sidebar.header("Regression Settings")
    independent_vars = st.sidebar.multiselect("Select independent variable(s) (X)", dataframe_model.columns.sort_values())
    dependent_var = st.sidebar.selectbox("Select dependent variable (Y)", dataframe_model.columns.sort_values())
    button_submit = st.sidebar.button("Start regression analysis")
    
    if independent_vars and dependent_var and button_submit:
        st.text('If you like the apps, kindly click share or star on my github') 
        st.text('-Indera') 
        regression_analysis(independent_vars, dependent_var, dataframe)
