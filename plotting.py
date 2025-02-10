import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from stock_utils import get_data
import os

def plot_stock_price(stock='SPY', start_date=None, end_date=None, n=10):
    """
    Fetches stock data and plots the closing price against time.
    """
    if start_date is None:
        start_date = datetime.today() - timedelta(days=365)  # Default to past year
    if end_date is None:
        end_date = datetime.today()
    
    # Fetch data
    data, idxs_with_mins, idxs_with_maxs = get_data(stock, start_date, end_date, n)
    
    # Plot data
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label=f'{stock} Closing Price', color='black')
    plt.scatter(data['Date'].iloc[idxs_with_mins], data['Close'].iloc[idxs_with_mins], color='green', label='Local Minima', marker='o')
    plt.scatter(data['Date'].iloc[idxs_with_maxs], data['Close'].iloc[idxs_with_maxs], color='red', label='Local Maxima', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.title(f'{stock} Closing Price Over Time')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)

    # Save the plot in the "results" folder
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    plot_path = os.path.join(results_folder, "data_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    plt.show()

if __name__ == "__main__":
    plot_stock_price()