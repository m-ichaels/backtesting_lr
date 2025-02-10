import yfinance as yf
from datetime import datetime, timezone, timedelta
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

def timestamp(dt):
    epoch = datetime.fromtimestamp(0, tz=timezone.utc)  # Corrected for timezone awareness
    return int((dt - epoch).total_seconds() * 1000)

def linear_regression(x, y):
    """
    performs linear regression given x and y. outputs regression coefficient
    """
    #fit linear regression
    lr = LinearRegression()
    lr.fit(x, y)

    return lr.coef_[0][0]

def n_day_regression(n, df, idxs):
    """
    n day regression.
    """
    #variable
    _varname_ = f'{n}_reg'
    df[_varname_] = np.nan

    for idx in idxs:
        if idx > n:
            
            y = df['Close'][idx - n: idx].to_numpy()
            x = np.arange(0, n)
            #reshape
            y = y.reshape(y.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            #calculate regression coefficient 
            coef = linear_regression(x, y)
            df.loc[idx, _varname_] = coef #add the new value
            
    return df

def normalized_values(high, low, close):
    """
    normalize the price between 0 and 1.
    """
    #epsilon to avoid deletion by 0
    epsilon = 10e-10
    
    #subtract the lows
    
    high = high - low
    close = close - low
    return close/(high + epsilon)

def get_stock_price(stock, date):
    """
    returns the stock price given a date
    """
    start_date = date - timedelta(days = 10)
    end_date = date
    data = yf.download(stock, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    
    return data['Close'].values[-1]

def get_data(stock, start_date = None, end_date = None, n = 10):
    data = yf.download(stock, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data =pd.DataFrame(data)

    #add the noramlzied value function and create a new column
    data['normalized_value'] = data.apply(lambda x: normalized_values(x['High'], x['Low'], x['Close']), axis = 1)
    #column with local minima and maxima
    data['loc_min'] = data.iloc[argrelextrema(data['Close'].values, np.less_equal, order = n)[0]]['Close']
    data['loc_max'] = data.iloc[argrelextrema(data['Close'].values, np.greater_equal, order = n)[0]]['Close']

    #idx with mins and max
    idx_with_mins = np.where(data['loc_min'] > 0)[0]
    idx_with_maxs = np.where(data['loc_max'] > 0)[0]
    
    return data, idx_with_mins, idx_with_maxs

def create_train_data(stock, start_date = None, end_date = None, n = 10):

    #get data to a dataframe
    data, idxs_with_mins, idxs_with_maxs = get_data(stock, start_date, end_date, n)
    
    #create regressions for 3, 5 and 10 days
    data = n_day_regression(3, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(5, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(10, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(20, data, list(idxs_with_mins) + list(idxs_with_maxs))
  
    _data_ = data[(data['loc_min'] > 0) | (data['loc_max'] > 0)].reset_index(drop = True)
    
    #create a dummy variable for local_min (0) and max (1)
    _data_['target'] = [1 if x > 0 else 0 for x in _data_['loc_max']]

    #columns of interest
    _data_ = _data_.rename(columns={'Close': 'close', 'Volume': 'volume'})
    cols_of_interest = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg', 'target']
    _data_ = _data_[cols_of_interest]

    return _data_.dropna(axis = 0)

def create_test_data_lr(stock, start_date = None, end_date = None, n = 10):
    """
    this function create test data sample for logistic regression model
    """
    #get data to a dataframe
    data, _, _ = get_data(stock, start_date, end_date, n)
    idxs = np.arange(0, len(data))
    
    #create regressions for 3, 5 and 10 days
    data = n_day_regression(3, data, idxs)
    data = n_day_regression(5, data, idxs)
    data = n_day_regression(10, data, idxs)
    data = n_day_regression(20, data, idxs)

    data = data.rename(columns={'Close': 'close', 'Volume': 'volume'})
    cols = ['close', 'volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']
    data = data[cols]
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    return data.dropna(axis = 0)

def predict_trend(stock, _model_, start_date = None, end_date = None, n = 10):

    #get data to a dataframe
    data, _, _ = get_data(stock, start_date, end_date, n)
    
    idxs = np.arange(0, len(data))
    #create regressions for 3, 5 and 10 days
    data = n_day_regression(3, data, idxs)
    data = n_day_regression(5, data, idxs)
    data = n_day_regression(10, data, idxs)
    data = n_day_regression(20, data, idxs)
        
    #create a column for predicted value
    data['pred'] = np.nan

    #get data
    data = data.rename(columns={'Close': 'close', 'Volume': 'volume'})
    cols = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']
    x = data[cols]
    x.columns = [col[0] if isinstance(col, tuple) else col for col in x.columns]

    #scale the x data
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    for i in range(x.shape[0]):
        
        try:
            data['pred'][i] = _model_.predict(x[i, :])

        except:
            data['pred'][i] = np.nan

    return data