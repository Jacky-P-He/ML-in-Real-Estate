import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Paths for models and metrics
model_path = 'models/'
metric_path = 'metrics/'

# Create directories if they don't exist
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(metric_path):
    os.makedirs(metric_path)

# Load data
@ st.cache_data
def load_data():
    train_data = pd.read_csv('data/train_data.csv')
    test_data = pd.read_csv('data/test_data.csv')
    return train_data, test_data

train_data, test_data = load_data()


data = st.session_state['processed_data']
mean_values_raw = data[['beds', 'full_baths', 'total_rooms']].mean()
std_devs_raw = data[['beds', 'full_baths', 'total_rooms']].std()

def save_model(model, model_type):
    joblib.dump(model, model_path + model_type + '.joblib')

def load_model(model_type):
    if os.path.exists(model_path + model_type + '.joblib'):
        return joblib.load(model_path + model_type + '.joblib')
    return None

def save_metrics(metrics, model_type):
    pd.DataFrame(metrics, index=[0]).to_csv(metric_path + model_type + '_metrics.csv', index=False)

def load_metrics():
    metrics = {}
    for file in os.listdir(metric_path):
        if file.endswith('_metrics.csv'):
            model_type = file.replace('_metrics.csv', '')
            metrics_df = pd.read_csv(metric_path + file)
            metrics[model_type] = metrics_df.iloc[0].to_dict()
    return metrics

def train_model(model_type):
    features = ['sqft_log', 'beds_std', 'full_baths_std', 'total_rooms_std', 'zip_code']
    target = 'sold_price_log'
    
    X_train = train_data[features]
    y_train = train_data[target].values
    
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=10)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)  # Predict on training set for simplicity
    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, y_pred)
    metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    save_model(model, model_type)
    save_metrics(metrics, model_type)
    return model_type + ' trained!'

def predict_price(model_type, input_features):
    model = load_model(model_type)
    if model:
        sqft_log = np.log(input_features[0] + 1)
        beds_std = (input_features[1] - mean_values_raw['beds']) / std_devs_raw['beds']
        full_baths_std = (input_features[2] - mean_values_raw['full_baths']) / std_devs_raw['full_baths']
        total_rooms_std = (input_features[3] - mean_values_raw['total_rooms']) / std_devs_raw['total_rooms']
        zip_code = input_features[4]

        transformed_features = [sqft_log, beds_std, full_baths_std, total_rooms_std, zip_code]
        prediction_log = model.predict([transformed_features])[0]
        prediction_price = np.exp(prediction_log)
        return f'Predicted Sold Price: ${prediction_price:.2f}'
    return 'Model not found.'


def main():
    st.title('House Price Prediction Models')
    
    model_type = st.selectbox('Choose a model to train:', ['Linear Regression', 'Random Forest', 'Gradient Boosting'])
    if st.button('Train Model'):
        result = train_model(model_type)
        st.success(result)
    
    if st.button('Display Metrics'):
        metrics = load_metrics()
        if metrics:
            st.write('Model Comparison:', pd.DataFrame(metrics).T)
        else:
            st.write("No models have been trained yet.")
    
    sqft = st.number_input('Enter square footage (sqft):', min_value=0)
    beds = st.number_input('Enter number of bedrooms (beds):', min_value=0)
    full_baths = st.number_input('Enter number of full bathrooms (full_baths):', min_value=0)
    total_rooms = st.number_input('Enter total number of rooms (total_rooms):', min_value=0)
    zip_code = st.number_input('Enter ZIP code (zip_code):', format='%d', value=10000)  # Default value to ensure integer input
    if st.button('Show Result'):
        input_features = [np.log(sqft + 1), beds, full_baths, total_rooms, zip_code]
        result = predict_price(model_type, input_features)
        st.write(result)

if __name__ == "__main__":
    main()



