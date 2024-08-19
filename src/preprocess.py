import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    """
    Preprocess stock data for modeling.
    
    :param data: Raw stock data DataFrame
    :return: Scaled DataFrame and the scaler used for scaling
    """
    # Only keep the 'Adj Close' column for prediction
    data = data[['Adj Close']]

    # Fill missing values
    data = data.copy()
    data.ffill(inplace=True)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Print the shape of the scaled data for debugging
    print(f"Scaled data shape: {scaled_data.shape}")
    
    return scaled_data, scaler

def create_sequences(data, sequence_length):
    """
    Create sequences of data for LSTM input.
    
    :param data: Scaled stock data
    :param sequence_length: Number of past time steps to consider for prediction
    :return: Arrays for LSTM input (X) and output (y)
    """
    X = []
    y = []
    
    # Print data shape for debugging
    print(f"Data shape before sequence creation: {data.shape}")

    # Loop through the data to create sequences
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])  # Input sequence
        y.append(data[i])  # Corresponding output value
    
    # Print the number of sequences created for debugging
    print(f"Number of sequences created: {len(X)}")

    # Convert to numpy arrays
    X, y = np.array(X), np.array(y)

    # Print shapes of X and y for debugging
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y
