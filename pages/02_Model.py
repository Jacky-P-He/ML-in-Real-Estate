import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  # For calculating RMSE

# Function to get model based on selection
def get_model(model_name, params):
    if model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Random Forest Regressor':
        model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    elif model_name == 'Gradient Boosting Regressor':
        model = GradientBoostingRegressor(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], max_depth=params['max_depth'])
    return model

def main():
    st.title('Model Training and Prediction')

    # Sidebar for choosing the model
    model_choice = st.selectbox('Select Model:', 
                                        ('Linear Regression', 'Random Forest Regressor', 'Gradient Boosting Regressor'))
    
    # Parameters based on model choice
    params = {}
    if model_choice == 'Random Forest Regressor' or model_choice == 'Gradient Boosting Regressor':
        params['n_estimators'] = st.sidebar.slider('Number of Trees (n_estimators):', 10, 500, 100)
        params['max_depth'] = st.sidebar.slider('Max Depth:', 1, 20, 5)
        if model_choice == 'Gradient Boosting Regressor':
            params['learning_rate'] = st.sidebar.slider('Learning Rate:', 0.01, 1.0, 0.1)

    # Data handling
    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']
        st.write("Data is loaded. Here's a sample:", data.head())
        X = data.drop('sold_price', axis=1)
        y = data['sold_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Button to train the model
        if st.button('Train Model'):
            model = get_model(model_choice, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            st.session_state['metrics'] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}
            st.write('Model trained successfully!')
            st.write(f'Mean Squared Error: {mse}')
            st.write(f'Root Mean Squared Error: {rmse}')
            st.write(f'R² Score: {r2}')

        # Making predictions
        if st.button('Make Prediction'):
            try:
                input_features = [float(x) for x in st.text_input('Enter feature values separated by comma:').split(',')]
                prediction = model.predict([input_features])[0]
                st.success(f'The predicted house price is ${prediction:.2f}')
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

        # Metric comparison
        metrics_to_compare = st.multiselect('Select metrics to compare:', ['MSE', 'RMSE', 'R2'], default=['MSE', 'R2'])
        if st.button('Compare Metrics'):
            if 'metrics' in st.session_state:
                for metric in metrics_to_compare:
                    st.write(f"{metric}: {st.session_state['metrics'][metric]}")
            else:
                st.error("No metrics to display. Please train the model first.")

    else:
        st.error("Data not loaded. Please go back to the Data Preprocessing page and upload data.")

if __name__ == '__main__':
    main()




