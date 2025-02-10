import pickle
import numpy as np
from stock_utils import create_test_data_lr, get_stock_price
from datetime import timedelta
import time
from sklearn.linear_model import LogisticRegression as lr

def load_LR(model_version):
    file = 'C:\\Users\\maver\\trade_wiv_me\\saved_models\\lr_v2.sav'
    loaded_model = pickle.load(open(file, 'rb'))
    return loaded_model

def load_scaler(model_version):
    file = 'C:\\Users\\maver\\trade_wiv_me\\saved_models\\scaler_v2.sav'
    loaded_model = pickle.load(open(file, 'rb'))
    return loaded_model

def _threshold(probs, threshold):
    """
    Inputs the probability and returns 1 or 0 based on the threshold.
    """
    prob_thresholded = [0 if x > threshold else 1 for x in probs[:, 0]]
    return np.array(prob_thresholded)

def LR_v1_predict(stock, start_date, end_date, threshold=0.99):
    """
    This function predicts given the data.
    Returns:
      - The raw prediction probability (as a numpy array),
      - The thresholded prediction,
      - The latest close price.
    """
    scaler = load_scaler('v2')
    model = load_LR('v2')

    data = create_test_data_lr(stock, start_date, end_date)
    close_price = data['close'].values[-1]

    input_data = data[['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']]
    input_data = input_data.to_numpy()[-1].reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)

    prediction_proba = model._predict_proba_lr(input_data_scaled)  # Use predict_proba() here
    prediction_thresholded = _threshold(prediction_proba, threshold)

    return prediction_proba[:, 0], prediction_thresholded[0], close_price

def LR_v1_sell(stock, buy_date, buy_price, todays_date, sell_perc=0.1, hold_till=3, stop_perc=0.05, sell_threshold=0.02):
    """
    Determines whether to sell the stock based on multiple conditions:
      - Current price falls below the stop loss price.
      - Current price exceeds the target sell price.
      - The holding period has elapsed.
      - **New Condition:** The model prediction probability (from a 40-day lookback) is below sell_threshold.
    
    Parameters:
      stock        - stock ticker symbol.
      buy_date     - the date the stock was bought.
      buy_price    - the price at which the stock was bought.
      todays_date  - current date.
      sell_perc    - percentage increase from buy_price to target sell price.
      hold_till    - number of days to hold the stock.
      stop_perc    - percentage decrease from buy_price to trigger stop loss.
      sell_threshold - if the prediction probability is below this value, then sell.
    
    Returns:
      A tuple (action, current_price), where action is either "SELL" or "HOLD".
    """
    current_price = get_stock_price(stock, todays_date)
    sell_price = buy_price + buy_price * sell_perc
    stop_price = buy_price - buy_price * stop_perc
    sell_date = buy_date + timedelta(days=hold_till)
    time.sleep(1)

    # New sell condition: Run the prediction model over the past 40 days.
    prediction, _, _ = LR_v1_predict(stock, todays_date - timedelta(days=40), todays_date)
    
    # Check sell conditions:
    # 1. Price falls below stop loss.
    # 2. Price meets or exceeds target sell price.
    # 3. Holding period has expired.
    # 4. Prediction probability is below sell_threshold.
    if (current_price is not None) and (
          (current_price < stop_price) or 
          (current_price >= sell_price) or 
          (todays_date >= sell_date) or 
          (prediction[0] < sell_threshold)
       ):
        return "SELL", current_price
    else:
        return "HOLD", current_price