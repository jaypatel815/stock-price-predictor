from src.data_loader import fetch_stock_data
from src.preprocess import preprocess_data, create_sequences
from src.model import train_lstm_model, make_predictions, make_future_predictions
from src.gui import run_gui
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def predict_stock(ticker):
    """
    Function to handle stock prediction.
    
    :param ticker: Stock ticker symbol
    """
    # Fetch data
    data = fetch_stock_data(ticker, '2021-01-01', '2024-08-16')

    # Ensure 'Date' is in the index and in datetime format
    data.index = pd.to_datetime(data.index)
    
    # Preprocess data
    scaled_data, scaler = preprocess_data(data)

    # Train model
    model, X_test, y_test = train_lstm_model(scaled_data)

    # Make predictions
    predictions = make_predictions(model, X_test, scaler)

    # Inverse transform the actual test data
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Get the dates corresponding to the test set (last len(y_test) dates)
    test_dates = data.index[-len(y_test):]

    # Plot predictions vs actual values
    fig, ax = plt.subplots()
    ax.plot(test_dates, y_test_inverse, label='Actual Price')
    ax.plot(test_dates, predictions, label='Predicted Price')

    # Generate future predictions
    num_future_dates = 30 # Number of days to predict into the future
    future_predictions = make_future_predictions(model, scaled_data, scaler, sequence_length=60, num_future_steps=num_future_dates)

    # Generate corresponding future dates
    last_date = test_dates[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=num_future_dates)

    # Plot future predictions
    ax.plot(future_dates, future_predictions, label='Future Predicted Price', linestyle='--', color='orange')

    # Set title and labels
    plt.title(f"Stock Price Prediciton for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")

    # Format the date on the x-axis
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))  # Set interval for ticks (e.g., every 10 days)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date to show year-month-day

    # Rotate the date labels for better readility
    plt.xticks(rotation=45)

    # Show legend and plot
    plt.legend()
    plt.show()
    
    # Display prediction (mockup for now)
    print(f"Prediction for {ticker}: [Mocked prediction]")

if __name__ == "__main__":
    run_gui(predict_stock)