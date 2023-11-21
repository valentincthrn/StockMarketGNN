import matplotlib.pyplot as plt
import torch
import pandas as pd
import streamlit as st


def plot_stock_predictions(historical_prices, predictions_tensor, timestamp_limit):
    historical_df = {
        ticker: pd.DataFrame(data).head(timestamp_limit)
        for ticker, data in historical_prices.items()
    }

    # Ensure the predictions tensor is converted to a numpy array for processing
    predictions = predictions_tensor.detach().numpy()

    # Plotting each company's data and prediction
    for i, (ticker, df) in enumerate(historical_df.items()):
        fig, ax = plt.subplots()
        ax.set_title(f"Stock Price and Predictions for {ticker}")

        # Plot historical prices
        ax.plot(df.index, df[(ticker, "price")], label="Historical Prices", linewidth=2)

        # Plot predictions
        predicted_prices = predictions[i]
        prediction_dates = pd.date_range(
            start=df.index[0], periods=len(predicted_prices) + 1, closed="right"
        )
        ax.plot(prediction_dates, predicted_prices, label="Predictions", linewidth=2)

        # Set background to transparent
        ax.set_facecolor("none")

        # Making the frame invisible
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Reducing the number of ticks and labels for a cleaner look
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        # Rotate date labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Making the grid lighter and set it to be behind the plots
        ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

        # Set legend
        ax.legend()

        # Remove the space between the subplots
        plt.tight_layout(pad=2)

        # Render the plot in Streamlit
        st.pyplot(fig)
