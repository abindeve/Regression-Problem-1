import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Function to rename columns for clarity
def rename_columns(df):
    df.columns = ['Sensor1', 'Sensor2', 'Target']
    return df

# Function to display sample data
def display_sample_data(df, title):
    st.write(f"## {title} Sample Data")
    st.dataframe(df.head())

# Function to upload CSV files and preprocess data
def upload_and_process_data():
    uploaded_train_file = st.sidebar.file_uploader("Upload training CSV file", type=["csv"])
    uploaded_test_file = st.sidebar.file_uploader("Upload testing CSV file", type=["csv"])

    if uploaded_train_file is not None and uploaded_test_file is not None:
        train_df = pd.read_csv(uploaded_train_file)
        test_df = pd.read_csv(uploaded_test_file)

        train_df = rename_columns(train_df)
        test_df = rename_columns(test_df)

        display_sample_data(train_df, "Training")
        display_sample_data(test_df, "Testing")

        return train_df, test_df
    else:
        return None, None

# Function to train linear regression model and calculate metrics
def train_and_evaluate_model(train_df, test_df):
    if train_df is not None and test_df is not None:
        # Separate features and target variable for both training and test datasets
        X_train = train_df[['Sensor1', 'Sensor2']]
        y_train = train_df['Target']
        X_test = test_df[['Sensor1', 'Sensor2']]
        y_test = test_df['Target']

        # Initialize and fit the linear regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # Predict on the test data
        y_pred = lr_model.predict(X_test)

        # Calculate MSE and MAE
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        return mse, mae
    else:
        return None, None

# Main function
def main():
    st.title("Linear Regression Model Evaluation")
    st.sidebar.title("Settings")

    train_df, test_df = upload_and_process_data()

    if train_df is not None and test_df is not None:
        mse, mae = train_and_evaluate_model(train_df, test_df)
        if mse is not None and mae is not None:
            st.write("## Model Evaluation Metrics")
            st.write(f"##### Mean Square Error MSE: {mse:.2f}")
            st.write(f"##### Mean Absolute Error MAE: {mae:.2f}")

if __name__ == "__main__":
    main()