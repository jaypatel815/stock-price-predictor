import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):

    """
    Fetch historical stock data using yfinance.
    
    :param ticker: Stock ticker symbol
    :param start_date: Start date for fetching data (format: 'YYYY-MM-DD')
    :param end_date: End date for fetching data (format: 'YYYY-MM-DD')
    :return: Pandas DataFrame with stock data
    """

    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data