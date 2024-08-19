import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.preprocess import create_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import Input

def build_lstm_model(input_shape):
    """
    Build and compile an improved LSTM model with dropout and multiple LSTM layers.
    
    :param input_shape: Shape of input data
    :return: Compiled LSTM model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # First LSTM layer with Dropout regularization
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Second LSTM layer with Dropout regularization
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    # Third LSTM layer
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    
    # Dense output layer
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model

def train_lstm_model(data, sequence_length=60, epochs=50, batch_size=32):
    """
    Train the LSTM model on the stock data with advanced features like early stopping and learning rate reduction.
    
    :param data: Preprocessed and scaled stock data
    :param sequence_length: The window size for LSTM input
    :param epochs: Number of training epochs
    :param batch_size: Batch size for training
    :return: Trained LSTM model and the test data
    """
    # Create sequences
    X, y = create_sequences(data, sequence_length)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Reshape the data for LSTM (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build the LSTM model
    model = build_lstm_model((X_train.shape[1], 1))

    # Set up callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-5)

    # Train the model
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(X_test, y_test), 
                        callbacks=[early_stopping, reduce_lr])
    
    # Plot training & validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    return model, X_test, y_test

def make_predictions(model, X_test, scaler):
    """
    Use the trained model to make predictions on the test data.

    :param model: Trained LSTM model
    :param X_test: Test data for prediction
    :param scaler: Fitted scaler to inverse transform the data
    :return: Inverse transformed predicted and actual values
    """
    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform the predictions back to original scale
    predictions = scaler.inverse_transform(predictions)

    return predictions

def make_future_predictions(model, data, scaler, sequence_length, num_future_steps):
    """
    Use the trained LSTM model to make future predictions beyond the test data.

    :param model: Trained LSTM model
    :param data: Preprocessed and scaled stock data
    :param scaler: Fitted scaler to inverse transform the data
    :param sequence_length: The window size for LSTM input
    :param num_future_steps: Number of future steps to predict
    :return: Future predicted values (in original scale)
    """
    # Get the last 'sequence_length' data points from the input data
    last_sequence = data[-sequence_length:].reshape(1, sequence_length, 1)

    future_predictions = []

    # Predict iteratively for 'num_future_steps' steps
    for _ in range(num_future_steps):
        # Make a prediction for the next step
        predicted_value = model.predict(last_sequence)
        
        # Store the predicted value
        future_predictions.append(predicted_value[0, 0])

        # Update the sequence: Remove the oldest value and add the new predicted value
        # Reshape the predicted_value to match the sequence shape (1, 1, 1)
        predicted_value = np.reshape(predicted_value, (1, 1, 1))
        last_sequence = np.append(last_sequence[:, 1:, :], predicted_value, axis=1)

    # Inverse transform the future predictions back to the original scale
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_inverse = scaler.inverse_transform(future_predictions)

    return future_predictions_inverse
