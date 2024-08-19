from src.data_loader import fetch_stock_data

def test_fetch():
    ticker = 'AAPL'  # Apple stock as an example
    start_date = '2024-08-10'
    end_date = '2024-08-16'

    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    # Print first few rows of the data
    print(stock_data.head())

if __name__ == "__main__":
    test_fetch()