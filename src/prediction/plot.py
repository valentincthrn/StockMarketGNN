import matplotlib.pyplot as plt
import torch
import pandas as pd
import streamlit as st


def plot_stock_predictions(historical_prices, predictions_tensor, timestamp_limit):
    # Convert the historical prices dictionary to a DataFrame for easier manipulation
    # The dictionary structure is assumed to be { 'ticker': {'quote_date': date, 'price': price} }

    st.write(historical_prices)

    historical_df = {
        ticker: pd.DataFrame(data.values(), index=data.keys()).tail(timestamp_limit)
        for ticker, data in historical_prices.items()
    }

    # Ensure the predictions tensor is converted to a numpy array for processing
    predictions = predictions_tensor.numpy()

    # Plotting each company's data and prediction
    for i, (ticker, df) in enumerate(historical_df.items()):
        plt.figure(figsize=(14, 7))
        plt.title(f"Stock Price and Predictions for {ticker}")

        # Plot historical prices
        plt.plot(df.index, df["price"], label="Historical Prices")

        # Plot predictions, assuming the last dimension of the tensor corresponds to the number of companies
        predicted_prices = predictions[:, i]
        prediction_dates = pd.date_range(
            start=df.index[-1], periods=len(predicted_prices) + 1, closed="right"
        )
        plt.plot(prediction_dates, predicted_prices, label="Predictions")

        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()
