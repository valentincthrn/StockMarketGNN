import matplotlib.pyplot as plt
import pandas as pd
from typing import List
import streamlit as st


def plot_training_pred(df_pred: pd.DataFrame, comps: List):
    # Set the style of matplotlib to 'ggplot' for better aesthetics
    plt.style.use("ggplot")

    fig, axs = plt.subplots(len(comps), 1, figsize=(10, 5 * len(comps)))

    # Loop through each subplot and plot the data
    for j, comp in enumerate(comps):
        # Setting the background color to transparent
        axs[j].set_facecolor("none")
        axs[j].plot(
            df_pred.index,
            df_pred[comp + "_pred"],
            label="pred",
            linewidth=2,
        )
        axs[j].plot(
            df_pred.index,
            df_pred[comp + "_true"],
            label="true",
            linewidth=2,
        )
        axs[j].set_title(comp)
        axs[j].legend()

        # Making the frame invisible
        for spine in axs[j].spines.values():
            spine.set_visible(False)

        # Reducing the number of ticks and labels for a cleaner look
        axs[j].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[j].yaxis.set_major_locator(plt.MaxNLocator(5))

        # Making the grid lighter and set it to be behind the plots
        axs[j].grid(
            True,
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7,
        )
        axs[j].set_axisbelow(True)

    # Remove the space between the subplots
    plt.tight_layout(pad=2)

    st.pyplot(fig)
