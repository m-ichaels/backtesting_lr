# Backtesting_lr

This project is designed to show my capability in developing a backtesting system for different machine learning algorithms. Showcased here is a "buy-low, sell-high" prediction tool (via logistic regression)

(Worth noting: I have also developed more accurate versions of this prediction model via Support Vector Machines and Neural Networks. Please get in touch for these more accurate models)

## Code Walkthrough

Training:

- Machine learning method: Logistic regression
- y-values: 'target' 10-day local minima and maxima
- First, stock data is sourced from yfinance (for 30ish stocks) data sourced is open, close, high, low, date, and volume. A dataframe is then made with the columns 'volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg' & 'target'. (Quick note: 'target' is made from the 10-day local minima and maxima prices, and is either 1, 0 or N/A.)
- This is then shuffled, and the target column is made the y-coordinate for logistic regression. The x-coordinate is the rest of the dataframe, with all of the values put through a MinMaxScaler to be between 0 and 1.
- The data is split, with 95% used for training and 5% used for testing.
- Logistic regression is then used, giving a 80ish percent accuracy of its predictions. Confusion matrices are produced, assessing the model's prediction accuracy.
- The model and scaler is then saved

---

Backtesting

- For a defined time period (currently using 1 year), the code scans through each stock listed on day 1, then 2, 3 etc.
- Whilst scanning, it gets the data for the stock's previous 40 days (to allow for full 20-day regression lines to be used), and prepares a scaled dataframe (same columns as above) to be fed into the model (obviously, not including the target value).
- It then makes a prediction, as to whether the the stock will go up or down (resulting in a value between 0 & 1, with 1 meaning it will go straight up and 2 meaning it will go straight down).
- This is then compared with the predicition threshold. This is done for all stocks on a particular day. Now, there are 3 possible scenarios:
  1. there are no predictions above the prediction threshold, in which case "No recommendations" is printed and we move onto the next day without buying any stocks.
  2. There is one stock above the threshold, in which case it is bought.
  3. There is multiple stocks above the prediction threshold, and the highest prediction number stock is used.
- In the case of buying stocks, the highest whole number of stocks are bought using the available cash. The buy price, numbe rof shares, total cost, and buy_date are all recorded. (there is an option to allocate a certain percentage of total capital to the order, but this is currently set as 1)
- Once a stock is bought, the is held until any 2 of these 4 conditions are met:
  1. if the current stock price is not 0 (should hopefully always be the case)
  2. if the current price is below the fixed stop loss (currently set as 0.3% below the buy price).
  3. if the current price is above the fixed sell price (currently set at 4% above the buy price).
  4. it has held the stock for longer than the maximum holding date (currently set at 21 days).

(there is an option to add a trailing stop loss, which I need to look into, and find the best way of doing that)

- Once the stock is sold, we wait until the next day and continue scanning for stocks to buy.

At the end, final loss/gain is tabulated.
