from src.data_loader import fetch_stock_data
from src.preprocess import preprocess_data, create_sequences

# Fetch some stock data
data = fetch_stock_data('AAPL', '2023-01-01', '2023-08-01')

# Preprocess the data
scaled_data, scaler = preprocess_data(data)

# Create sequences with a sequence length of 60
sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)
