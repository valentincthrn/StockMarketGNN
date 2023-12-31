import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from uniplot import plot
from io import BytesIO
from pandas.tseries.offsets import BDay
from pathlib import Path

def plot_stock_predictions(
    historical_prices, predictions_tensor, timestamp_limit, comps, last_raw_price, df_prices_raw, subfolder_name
):

    historical_df = pd.concat(historical_prices.values(), axis=1)
    historical_df = historical_df.shift(1)
    historical_df.columns = [c[0] for c in historical_df.columns]

    for col in historical_df.columns:
        historical_df[col].iloc[0] = last_raw_price[col]
    historical_df.iloc[1:, :] = (historical_df.iloc[1:, :] + 1).cumprod(axis=0)
    historical_df.iloc[1:, :] = historical_df.iloc[0] / historical_df.iloc[1:, :]

    # Ensure the predictions tensor is converted to a numpy array for processing
    predictions = predictions_tensor.detach().cpu().numpy()
    df_predictions = pd.DataFrame(dict(zip(comps, predictions)))

    # Compute the next N business days from the last date of the existing index
    horizon = len(df_predictions)
    next_business_days = pd.date_range(
        start=historical_df.index[0] + BDay(1), periods=horizon, freq=BDay()
    )
    df_predictions.index = next_business_days

    df_hist_with_pred = pd.concat([df_predictions, historical_df], axis=0).sort_index()

    df_hist_with_pred.iloc[-horizon:, :] = (
        df_hist_with_pred.iloc[-horizon:, :] + 1
    ).values
    df_hist_with_pred.iloc[-horizon - 1 :] = (
        df_hist_with_pred.iloc[-horizon - 1 :].cumprod(axis=0).values
    )
    
    snapshot = max(historical_df.index)
    df_prices_raw.columns = [c[0]+ "_true" for c in df_prices_raw.columns]
    df_prices_raw = df_prices_raw.loc[lambda f: f.index >= snapshot]
    snapshot_str = snapshot.strftime("%Y-%m-%d")
    
    # Check if df_prices_raw has more than one row for merging
    if len(df_prices_raw) > 1:
        df = df_hist_with_pred.merge(df_prices_raw, left_index=True, right_index=True, how="left")
    else:
        df = df_hist_with_pred.copy()

    # IMPORTANT: Cache the conversion to prevent computation on every rerun

    # Generate the CSV file in memory
    csv = df.to_csv().encode('utf-8')
    csv_bytesIO = BytesIO(csv)

    # The actual download button
    st.download_button(
        label="Download data as CSV",
        data=csv_bytesIO,
        file_name=f"pred_and_history_{snapshot_str}.csv",
        mime='text/csv',
    )
    # Plotting each company's data and prediction
    for ticker in df_hist_with_pred.columns:
        fig, ax = plt.subplots()
        ax.set_title(f"Stock Price and Predictions for {ticker}")

        ts = df_hist_with_pred[[ticker]]
        # if ticker == "bpac11":
        #     st.dataframe(ts)
        history = ts.loc[lambda f: f.index <= historical_df.index[0]]
        horizon = ts.loc[lambda f: f.index >= historical_df.index[0]]
        if len(df_prices_raw) > 1:
            horizon_true = df_prices_raw[[ticker+"_true"]]
            ax.plot(horizon_true, label="Futures Prices", linewidth=2, color="b", linestyle='dotted')

        # Plot historical prices
        ax.plot(history, label="Historical Prices", linewidth=2, color="b")

        ax.plot(horizon, label="Predictions", linewidth=2, color = "r", linestyle='dashed')

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


def plot_uniplot(df_pred_prices, comps):

    for comp in comps:
        df_to_plot = df_pred_prices[[comp + "_pred", comp + "_true"]]
        plot(df_to_plot.values.T, lines=True, title=f"{comp} Predictions vs True")
