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
    Inputs the probability and returns 1 or 0 based on the threshold
    """
    prob_thresholded = [0 if x > threshold else 1 for x in probs[:, 0]]

    return np.array(prob_thresholded)

def LR_v1_predict(stock, start_date, end_date, threshold = 0.98):
    """
    This function predicts given the data
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

def LR_v1_sell(stock, buy_date, buy_price, todays_date, sell_perc = 0.1, hold_till = 3, stop_perc = 0.05):
    """
    gets stock price. Recommend to sell if the stock price is high sell_perc * buy_price + buy_price
    stock - stock ticker symbol
    buy_date - the date the stock was bought
    todays_date - date today
    sell_perc - sell percentage
    hold_till - how many days to hold from today
    """
    current_price = get_stock_price(stock, todays_date)
    sell_price = buy_price + buy_price * sell_perc
    stop_price = buy_price - buy_price * stop_perc
    sell_date = buy_date + timedelta(days = hold_till)
    time.sleep(1)

    if (current_price is not None) and ((current_price < stop_price) or (current_price >= sell_price) or (todays_date >= sell_date)):
        return "SELL", current_price
    else:
        return "HOLD", current_price