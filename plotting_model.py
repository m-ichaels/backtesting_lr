import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from lr_inference import LR_v1_predict
from tqdm import tqdm
import numpy as np

def main():
    # Configuration parameters
    stock = "EURGBP=X"
    threshold = 0.92       # threshold parameter to pass to LR_v1_predict
    sell_threshold = 0.08  # threshold for sell signal
    back_to_days = 40      # number of days to look back for the model input
    start_date = datetime(2024, 4, 1)
    end_date = datetime(2025, 4, 1)

    # Lists to hold data for plotting
    dates = []
    close_prices = []
    buy_signal_dates = []
    buy_signal_prices = []
    sell_signal_dates = []
    sell_signal_prices = []

    total_days = (end_date - start_date).days
    current_day = start_date

    # Loop through each day in the date range
    for _ in tqdm(range(total_days), desc="Processing Days"):
        try:
            # Define the historical period for the model input
            period_start = current_day - timedelta(days=back_to_days)
            # Get the model output for the given day
            prediction, prediction_thresholded, close_price = LR_v1_predict(
                stock, period_start, current_day, threshold
            )
            # Record the close price for this day
            dates.append(current_day)
            close_prices.append(close_price)

            # Check if a buy signal is generated.
            # In your model a buy signal is when prediction_thresholded is less than 1.
            if prediction_thresholded < 1:
                buy_signal_dates.append(current_day)
                buy_signal_prices.append(close_price)
            
            # Check if a sell signal is generated.
            # We mark a sell signal if the raw prediction is below sell_threshold.
            if prediction < sell_threshold:
                sell_signal_dates.append(current_day)
                sell_signal_prices.append(close_price)

        except Exception as e:
            print(f"Error processing {current_day.strftime('%Y-%m-%d')}: {e}")
            # Append NaN if there was an error (e.g. missing data on non-trading days)
            dates.append(current_day)
            close_prices.append(np.nan)

        # Move to the next day
        current_day += timedelta(days=1)

    # Create the plot
    plt.figure(figsize=(14, 7))
    plt.plot(dates, close_prices, label='Close Price', color='black')
    plt.scatter(buy_signal_dates, buy_signal_prices, color='green', label='Buy Signal', zorder=5)
    plt.scatter(sell_signal_dates, sell_signal_prices, color='red', label='Sell Signal', zorder=5)
    
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'{stock} Close Price vs Time with Buy and Sell Signals')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot in the "results" folder
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    plot_path = os.path.join(results_folder, "model_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Optionally, display the plot (remove if not needed)
    plt.show()

if __name__ == "__main__":
    main()