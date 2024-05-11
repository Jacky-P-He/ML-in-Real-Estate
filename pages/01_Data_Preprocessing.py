import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder

def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

def log_transform(feature):
    return np.log1p(feature)  # avoid log(0)


def standardize(feature):
    mean_val = feature.mean()
    std_val = feature.std()
    return (feature - mean_val) / std_val


def normalize(feature):
    min_val = feature.min()
    max_val = feature.max()
    return (feature - min_val) / (max_val - min_val)


def create_feature(df, operation, feature_list, new_feature_name):
    if operation == 'add':
        df[new_feature_name] = df[feature_list].sum(axis=1)
    elif operation == 'subtract':
        df[new_feature_name] = df[feature_list[0]] - df[feature_list[1]]
    elif operation == 'multiply':
        df[new_feature_name] = df[feature_list[0]] * df[feature_list[1]]
    elif operation == 'divide':
        df[new_feature_name] = df[feature_list[0]] / df[feature_list[1]]
    return df


def compute_descriptive_stats(df, features):
    stats = {}
    for feature in features:
        stats[feature] = {
            'mean': df[feature].mean(),
            'median': df[feature].median(),
            'min': df[feature].min(),
            'max': df[feature].max()
        }
    return stats


def compute_correlation(df, features):
    correlation = df[features].corr()
    correlation_summary = []
    for f1 in features:
        for f2 in features:
            if f1 != f2:
                cor = correlation[f1][f2]
                summary = f"- Features {f1} and {f2} are {'strongly' if abs(cor) > 0.5 else 'weakly'} {'positively' if cor > 0 else 'negatively'} correlated: {cor:.2f}"
                correlation_summary.append(summary)
    return correlation, correlation_summary


def main():
    st.title('NYC House Data Preprocessing and Visualization')
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        housing_data = pd.read_csv(uploaded_file)
        st.write("Original Data Sample:", housing_data.head())
        st.markdown("""
        ## Data Cleaning and Feature Engineering Steps

        The following steps were performed to clean the data and prepare it for further analysis:

        1. **Handle Missing Values:**
        - Fill missing values in 'half_baths' with 0.
        - Fill missing values in 'sqft' with the median value of the column.
        - Compute 'price_per_sqft' as 'sold_price' divided by 'sqft' where 'price_per_sqft' is missing.

        2. **Drop Unnecessary Columns:**
        - Remove columns that are not relevant to the analysis such as 'days_on_mls', 'assessed_value', 'estimated_value', 'lot_sqft', and 'stories'.

        3. **Geographical Adjustments:**
        - For missing 'latitude' and 'longitude' within each 'zip_code', fill with the median value of existing entries.

        4. **Fill Missing Values for Key Demographic Features:**
        - Fill missing values for 'full_baths', 'beds', 'year_built', and 'list_price' using the median of each column.

        5. **Adjustment of Zip Code and County:**
        - Fill missing 'zip_code' and 'county' using the mode (most frequent value) of each column.
        - Ensure 'zip_code' is treated as an integer.

        6. **Feature Engineering:**
        - Calculate 'house_age' by subtracting 'year_built' from the current year (2024).
        - Encode categorical 'style' feature using one-hot encoding, dropping the first category to avoid multicollinearity.

        7. **Data Transformation:**
        - Apply logarithmic transformation to 'sqft', 'list_price', and 'sold_price' to normalize their distribution.
        - Standardize 'house_age', 'beds', 'full_baths', and 'half_baths' to have mean zero and standard deviation of one.
        - Normalize newly created one-hot encoded style features to range between 0 and 1.

        8. **Create Combined Features:**
        - Create a 'total_rooms_std' feature by adding standardized 'beds_std', 'full_baths_std', and 'half_baths_std'.
        - Compute 'price_per_sqft_log' by dividing 'list_price_log' by 'sqft_log'.
        """)


        housing_data['half_baths'].fillna(0, inplace=True)
        housing_data.drop(columns=['days_on_mls', 'assessed_value', 'estimated_value', 'lot_sqft', 'stories'], inplace=True)
        housing_data['sqft'].fillna(housing_data['sqft'].median(), inplace=True)
        housing_data.loc[housing_data['price_per_sqft'].isnull(), 'price_per_sqft'] = housing_data['sold_price'] / housing_data['sqft']
        
        median_values = {
            'full_baths': housing_data['full_baths'].median(),
            'beds': housing_data['beds'].median(),
            'year_built': housing_data['year_built'].median(),
            'list_price': housing_data['list_price'].median()
        }
        housing_data.fillna(median_values, inplace=True)

        for zip_code in housing_data['zip_code'].unique():
            lat_median = housing_data.loc[housing_data['zip_code'] == zip_code, 'latitude'].median()
            long_median = housing_data.loc[housing_data['zip_code'] == zip_code, 'longitude'].median()
            housing_data.loc[(housing_data['zip_code'] == zip_code) & (housing_data['latitude'].isnull()), 'latitude'] = lat_median
            housing_data.loc[(housing_data['zip_code'] == zip_code) & (housing_data['longitude'].isnull()), 'longitude'] = long_median

        housing_data['zip_code'].fillna(housing_data['zip_code'].mode()[0], inplace=True)
        housing_data['county'].fillna(housing_data['county'].mode()[0], inplace=True)
        housing_data.dropna(subset=['latitude', 'longitude'], inplace=True)
        housing_data['zip_code'] = housing_data['zip_code'].astype(int)


        current_year = 2024
        housing_data['house_age'] = current_year - housing_data['year_built']
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        style_encoded = encoder.fit_transform(housing_data[['style']])
        style_encoded_df = pd.DataFrame(style_encoded, columns=encoder.get_feature_names_out())
        housing_data = pd.concat([housing_data.reset_index(drop=True), style_encoded_df], axis=1)

        for feature in ['sqft', 'list_price', 'sold_price']:
            housing_data[feature + '_log'] = log_transform(housing_data[feature])

        for feature in ['house_age', 'beds', 'full_baths', 'half_baths']:
            housing_data[feature + '_std'] = standardize(housing_data[feature])

        for col in [col for col in housing_data.columns if col.startswith('style_')]:
            housing_data[col + '_norm'] = normalize(housing_data[col])


        housing_data = create_feature(housing_data, 'add', ['beds_std', 'full_baths_std', 'half_baths_std'], 'total_rooms_std')
        housing_data = create_feature(housing_data, 'divide', ['list_price_log', 'sqft_log'], 'price_per_sqft_log')


        selected_features = ['style_CONDO_norm', 'style_CONDOS_norm', 'style_COOP_norm', 'style_LAND_norm', 'style_MOBILE_norm', 'style_MULTI_FAMILY_norm', 'style_OTHER_norm', 'style_SINGLE_FAMILY_norm', 'style_TOWNHOMES_norm', 'sqft_log', 'list_price_log', 'sold_price_log', 'house_age_std', 'beds_std', 'full_baths_std', 'half_baths_std', 'total_rooms_std', 'price_per_sqft_log']
        descriptive_stats = compute_descriptive_stats(housing_data, selected_features)
        correlation_matrix, correlation_summary = compute_correlation(housing_data, selected_features)
        st.session_state['processed_data'] = housing_data
        st.write("Processed Data Sample:", housing_data.head())

        options = [
            'Scatter Matrix', 
            'Lineplot', 
            'Histogram', 
            'Boxplot',
            'Descriptive Statistics',
            'Correlation Matrix'
        ]
        
        selected_options = st.multiselect('Choose visualizations to display:', options, default=options)
        features = list(housing_data.columns)

        if 'Descriptive Statistics' in selected_options:
            st.subheader('Descriptive Statistics')
            descriptive_stats = compute_descriptive_stats(housing_data, selected_features)
            st.dataframe(descriptive_stats)

        if 'Correlation Matrix' in selected_options:
            st.subheader('Correlation Matrix')
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(correlation_matrix, cmap='coolwarm', ax=ax)
            plt.title('Correlation Matrix')
            st.pyplot(fig)

        if 'Scatter Matrix' in selected_options:
            st.subheader('Scatter Matrix of Selected Features')
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter_matrix(housing_data[['sqft_log', 'list_price_log', 'sold_price_log', 'house_age_std', 'beds_std', 'full_baths_std', 'total_rooms_std', 'price_per_sqft_log']], 
                           alpha=0.7, diagonal='hist', color='blue', ax=ax)
            plt.suptitle('Scatter Matrix of Selected Features', size=16)
            st.pyplot(fig)

        if 'Lineplot' in selected_options:
            st.subheader('Lineplot')
            x_axis = st.selectbox('Choose X axis:', features, key='x_axis_line')
            y_axis = st.selectbox('Choose Y axis:', features, key='y_axis_line')
            fig, ax = plt.subplots()
            sns.lineplot(x=x_axis, y=y_axis, data=housing_data, ax=ax)
            ax.set_title(f'{y_axis} vs. {x_axis}')
            st.pyplot(fig)

        if 'Histogram' in selected_options:
            st.subheader('Histogram')
            x_axis = st.selectbox('Choose a variable for Histogram:', features, key='x_axis_hist')
            fig, ax = plt.subplots()
            sns.histplot(housing_data[x_axis], kde=True, ax=ax)
            ax.set_title(f'Distribution of {x_axis}')
            st.pyplot(fig)

        if 'Boxplot' in selected_options:
            st.subheader('Boxplot')
            x_axis = st.selectbox('Choose X axis (categorical):', features, key='x_axis_box')
            y_axis = st.selectbox('Choose Y axis:', features, key='y_axis_box')
            fig, ax = plt.subplots()
            sns.boxplot(x=x_axis, y=y_axis, data=housing_data, ax=ax)
            ax.set_title(f'{y_axis} vs. {x_axis}')
            st.pyplot(fig)

if __name__ == '__main__':
    main()