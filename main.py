from pandas.api.types import is_numeric_dtype
import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random
from dateutil import parser
import subprocess
from dateutil.parser import parse

df = pd.read_csv('BatteryModules2.csv', encoding='unicode_escape')
date_formats = ['%dth %B', '%d-%m-%Y', '%Y-%m-%d']

st.title("Data Exploration and Analysis")

name_stat1 = st.selectbox("Select the name for the Column Selection - 1:", df.columns.unique(), key="name_stat1")
if name_stat1 not in df.columns.values:
    st.error("Wrong input. column name not found.")
name_stat2 = st.selectbox("Select the name for the Column Selection - 2:", df.columns.unique(), key="name_stat2")
if name_stat2 not in df.columns.values:
    st.error("Wrong input. column name not found.")

tog_button = st.checkbox("Execute profiling?")
if tog_button:

    missing_values = df.isnull().sum()
    total_values = df.shape[0]

    # Calculate the percentage of null and non-null values
    null_percent = (missing_values / total_values) * 100

    st.subheader("Completeness Check")
    st.write("Percentage of Null Values:")
    st.write(null_percent)

    # Group values by the specified column
    group_vals = df[name_stat1].value_counts()
    st.subheader(f"Grouped Values of {name_stat1}")
    st.write(group_vals)

    group_vals = df[name_stat2].value_counts()
    st.subheader(f"Grouped Values of {name_stat2}")
    st.write(group_vals)


    def per(name_stat1):
        quantiles = df[name_stat1].quantile([0.05, 0.95])
        value_range = quantiles.values
        count = 0
        rejected = []
        actual = df[name_stat1].tolist()

        for i in actual:
            i = float(i)
            if i >= value_range[0] and i <= value_range[1]:
                count += 1
            else:
                rejected.append(i)

        data = {
            f"Statistic for {name_stat1}": ["Range of values", "Count in range", "Total count"],
            "Value": [f"{value_range}", count, len(actual)]
        }
        df1 = pd.DataFrame(data)

        st.table(df1)


    if is_numeric_dtype(df[name_stat1]):
        per(name_stat1)
    if is_numeric_dtype(df[name_stat2]):
        per(name_stat2)


    # report = create_report(df)

    def encoding(pass_list):
        encoded_pattern = []

        def get_text_format(pass_list):
            for i in pass_list:
                result = ''
                for char in str(i):
                    if char.isalpha():
                        result += 'x'
                    elif char.isdigit():
                        result += 'n'
                    else:
                        result += char
                encoded_pattern.append(result)

        get_text_format(pass_list)
        final_dict = {}

        for i in encoded_pattern:
            if i not in final_dict:
                final_dict[i] = 1
            else:
                final_dict[i] += 1

        final_pd = pd.DataFrame.from_dict([final_dict])
        final_pd.head()
        return final_pd


    def classify(pass_dict1, pass_dict2):
        count_aplha1 = 0
        count_num1 = 0
        count_aplha2 = 0
        count_num2 = 0

        for key, value in pass_dict1.items():
            key = str(key)  # Convert the key to a string
            if 'x' in key and 'n' not in key:
                count_aplha1 += value
            elif 'n' in key and 'x' not in key:
                count_num1 += value

        for key, value in pass_dict2.items():
            key = str(key)  # Convert the key to a string
            if 'x' in key and 'n' not in key:
                count_aplha2 += value
            elif 'n' in key and 'x' not in key:
                count_num2 += value

        st.info(f"The number of strings in {name_stat1} column only: {count_aplha1}")
        st.info(f"The number of numerical values in {name_stat1} column only: {count_num1}")
        st.info(f"The number of strings in {name_stat2} column only: {count_aplha2}")
        st.info(f"The number of numerical values in {name_stat2} column only: {count_num2}")


    pass_list = df[name_stat1].tolist()
    final_pd = encoding(pass_list)
    # final_pd.insert(0, "Description", "count of total values")

    # Display the encoded result with the descriptive column
    st.subheader(f"Pattern Result of the {name_stat1} column")
    st.write(final_pd)

    # Repeat the above steps for the second column
    pass_list2 = df[name_stat2].tolist()
    final_pd1 = encoding(pass_list2)
    # final_pd1.insert(0, "Description", "count of total values")
    st.subheader(f"Encoding Result of the {name_stat2} column")
    st.write(final_pd1)
    keymax1 = final_pd.columns[np.argmax(final_pd.iloc[0].values)]
    st.write(f"The accepted format for {name_stat1} is:", keymax1, final_pd.iloc[0][keymax1])

    keymax2 = final_pd1.columns[np.argmax(final_pd1.iloc[0].values)]
    st.write(f"The accepted format for {name_stat2} is:", keymax2, final_pd1.iloc[0][keymax2])
    classify(final_pd.iloc[0].to_dict(), final_pd1.iloc[0].to_dict())


    def find_expected_values(choice, final_pd_0):
        final_val = list(final_pd_0.iloc[0].values)
        final_val.sort()
        expected_val = []

        for i in final_val:
            if (final_val[len(final_val) - 1] - i) <= choice and i != final_val[len(final_val) - 1]:
                expected_val.append(i)

        return expected_val


    pass_list = df[name_stat1].tolist()
    final_pd = encoding(pass_list)

    # ... (Rest of the code remains unchanged) ...

    st.subheader(f"Pattern Result of the {name_stat1} column")
    st.write(final_pd)

    # Display the encoded result with a dropdown for user selection
    st.subheader(f"Choose from Encoded Options for {name_stat1}")
    selected_encoding = st.selectbox(f"Select an encoded option for {name_stat1}", final_pd.columns)

    # Get the total number of appearances for each encoding pattern
    encoding_counts = final_pd.iloc[0].to_dict()
    st.subheader("Total Appearances for Each Encoding Pattern")
    st.write(encoding_counts)

    keymax1 = final_pd.columns[np.argmax(final_pd.iloc[0].values)]
    st.write(f"The accepted format for {name_stat1} is:", keymax1, final_pd.iloc[0][keymax1])


    if selected_encoding:
        st.subheader(f"Your Chosen Encoding Option for {name_stat1}")
        st.write(f"Chosen Option: {selected_encoding}")

    pass_list1 = df[name_stat2].tolist()
    final_pd1 = encoding(pass_list1)
    st.subheader(f"Pattern Result of the {name_stat2} column")
    st.write(final_pd1)
    st.subheader(f"Choose from Encoded Options for {name_stat2}")
    selected_encoding = st.selectbox(f"Select an encoded option for {name_stat2}", final_pd1.columns)
    encoding_counts = final_pd1.iloc[0].to_dict()
    st.subheader("Total Appearances for Each Encoding Pattern")
    st.write(encoding_counts)
    keymax2 = final_pd1.columns[np.argmax(final_pd1.iloc[0].values)]
    st.write(f"The accepted format for {name_stat2} is:", keymax2, final_pd1.iloc[0][keymax2])
    if selected_encoding:
        st.subheader(f"Your Chosen Encoding Option for {name_stat2}")
        st.write(f"Chosen Option: {selected_encoding}")



    def separate_date_and_time(input_list):
        df1 = pd.DataFrame(columns=['Date', 'Time'])
        for entry in input_list:
            if isinstance(entry, str):
                try:
                    # Parse the input string to identify the date and time components
                    dt = parse(entry)
                    date_str = dt.date()
                    time_str = dt.time()
                    df1 = df1.append({'Date': date_str, 'Time': time_str}, ignore_index=True)
                except ValueError:
                    print(f"Warning: Unable to parse '{entry}'. Skipping this entry.")
            else:
                print(f"Warning: Input '{entry}' is not a string. Skipping this entry.")

        return df1
    name_stat4 = st.selectbox("Select the date column to  be transformed :",df.columns.unique() , key="name_stat4")
    if name_stat4 not in df.columns.values:
        st.error("Wrong input. column name not found.")
    else:
        df1 = separate_date_and_time(df[name_stat4][::20])
        df['New_Date'] = df1['Date']
        df['New_Time'] = df1['Time']
    st.write(df.head(10))
    def convert_date(name_stat3):
            target_date_format = name_stat3

            # Convert values in 'New_Date' column to datetime objects
            df['New_Date'] = pd.to_datetime(df['New_Date'], errors='coerce')

            # Filter out any NaT (Not a Time) values
            valid_dates = df['New_Date'].dropna()

            # Convert valid dates to strings with the chosen format
            converted_dates = [date.strftime(target_date_format) for date in valid_dates]

            # Update the 'ConvertedDate' column
            df['ConvertedDate'] = [None] * len(df)  # Initialize with None
            df.loc[df['New_Date'].notnull(), 'ConvertedDate'] = converted_dates

            # Display the updated DataFrame
            st.subheader("Updated DataFrame")


    name_stat3 = st.selectbox("Select the format for formatting the requested date column :", date_formats, key="name_stat3")
    convert_date(name_stat3)
    st.write(df['ConvertedDate'].head(10))



    if is_numeric_dtype(df[name_stat1]) and is_numeric_dtype(df[name_stat2]):

        top_values_y = st.selectbox("Select the number of top values for y-axis", [5, 10, 30, 50])

        # Sort the DataFrame based on the specified column names
        sorted_df = df.sort_values(by=[name_stat1, name_stat2])

        # Get the unique values for x-axis
        unique_values_x = sorted_df[name_stat1].unique()

        # Get the counts for the unique values in column name_stat2
        unique_values_counts = sorted_df[name_stat2].value_counts()

        # Select the top values for y-axis based on their counts
        top_df_y = unique_values_counts.nlargest(top_values_y).index

        # Create a line chart with the top values
        st.line_chart(data=df[df[name_stat2].isin(top_df_y)], x=name_stat1, y=name_stat2, width=0, height=0,
                      use_container_width=True)
    else:
        top_values_y = st.selectbox("Select the number of top values for y-axis", [5, 10, 30, 50])

        # Sort the DataFrame based on the specified column names
        sorted_df = df.sort_values(by=[name_stat1, name_stat2])

        # Get the unique values for x-axis
        unique_values_x = sorted_df[name_stat1].unique()

        # Get the counts for the unique values in column name_stat2
        unique_values_counts = sorted_df[name_stat2].value_counts()

        # Select the top values for y-axis based on their counts
        top_df_y = unique_values_counts.nlargest(top_values_y).index

        # Create a bar chart with the top values
        st.bar_chart(data=df[df[name_stat2].isin(top_df_y)], x=name_stat1, y=name_stat2, width=0, height=0,
                     use_container_width=True)

    if st.button("Launch App"):
        subprocess.run(["streamlit", "run", "app.py"])
