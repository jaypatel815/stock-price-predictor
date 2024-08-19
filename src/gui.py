import tkinter as tk
from tkinter import ttk

def run_gui(predict_callback):
    """
    GUI for stock price predictor.
    
    :param predict_callback: Function to call when prediction is triggered
    """
    root = tk.Tk()
    root.title('Stock Price Predictor')

    # Stock ticker entry
    tk.Label(root, text="Stock Ticker:").grid(row=0, column=0)
    ticker_entry = tk.Entry(root)
    ticker_entry.grid(row=0, column=1)

    # Prediction button
    predict_button = ttk.Button(root, text="Predict", command=lambda: predict_callback(ticker_entry.get()))
    predict_button.grid(row=1, columnspan=2)

    root.mainloop()