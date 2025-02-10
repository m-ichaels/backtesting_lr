from simulator import simulator
import os
from datetime import timedelta, datetime
from lr_inference import LR_v1_predict, LR_v1_sell
from tqdm import tqdm
from collections import OrderedDict
import pickle
import numpy as np

class backtester(simulator):

    def __init__(self, stocks_list, model, capital, start_date, end_date, threshold = 0.99, sell_perc = 0.04, hold_till = 21, stop_perc = 0.003, sell_threshold = 0.02):

        super().__init__(capital)

        self.stocks = stocks_list
        self.model = model
        self.start_date = start_date
        self.day = start_date
        self.end_date = end_date
        self.status = 'buy'
        self.threshold = threshold
        self.sell_perc = sell_perc
        self.hold_till = hold_till
        self.stop_perc = stop_perc
        self.sell_threshold = sell_threshold

        current_dir = os.getcwd()
        results_dir = os.path.join(current_dir, 'results')
        folder_name = f'{str(self.model.__name__)}_{self.threshold}_{self.hold_till}'
        self.folder_dir = os.path.join(results_dir, folder_name)
        if not os.path.exists(self.folder_dir):
            os.makedirs(self.folder_dir)

    def backtest(self):
        """
        start backtesting
        """
        delta = timedelta(days = 1)

        total_days = (self.end_date - self.start_date).days
        d = 0
        pbar = tqdm(total = total_days)

        while self.day <= self.end_date:

            self.daily_scanner = {}
            if self.status == 'buy':
                self.scanner()
                if list(self.daily_scanner.keys()) != []:
                    recommended_stock = list(self.daily_scanner.keys())[0]
                    recommended_price = list(self.daily_scanner.values())[0][2]
                    self.buy(recommended_stock, recommended_price, self.day)
                    print(f'Bought {recommended_stock} for {recommended_price} on the {self.day}')
                    self.status = 'sell'
                else:
                    print('No recommendations')
            else:
                stocks = [key for key in self.buy_orders.keys()]
                for s in stocks:
                    recommended_action, current_price = LR_v1_sell(s, self.buy_orders[s][3], self.buy_orders[s][0], self.day, self.sell_perc, self.hold_till, self.stop_perc, self.sell_threshold)
                if recommended_action == "SELL":
                    print(f'Sold {s} for {current_price} on {self.day}')
                    self.sell(s, current_price, self.buy_orders[s][1], self.day)    
                    self.status = 'buy'
            self.day += delta
            d+= 1
            pbar.update(1)
        pbar.close()
        self.print_bag()
        self.print_summary()
        self.save_results()
        return
        
    def get_stock_data(self, stock, back_to = 40):
        """
        this function queries yfinance and gets data of a particular stock on a given day back to a certain number of days (default is 30)
        """
        end = self.day
        start = self.day - timedelta(days = back_to)
        prediction, prediction_thresholded, close_price = self.model(stock, start, end, self.threshold)
        return prediction[0], prediction_thresholded, close_price
        
    def scanner(self):
        """
        scan the stocks to find good stocks
        """
        for stock in self.stocks:
            try:
                prediction, prediction_thresholded, close_price = self.get_stock_data(stock)
                print(f'Stock: {stock}, Prediction: {prediction}, Thresholded: {prediction_thresholded}, Close Price: {close_price}')

                if prediction_thresholded < 1:
                    self.daily_scanner[stock] = (prediction, prediction_thresholded, close_price)
            except Exception as e:
                print(f'Error fetching data for {stock}: {e}')
                pass

        def take_first(elem):
            return elem[1]

        self.daily_scanner = OrderedDict(sorted(self.daily_scanner.items(), key=take_first, reverse=True))

    def save_results(self):
        """
        save history dataframe create figures and save
        """
        results_df_path = os.path.join(self.folder_dir, 'history_df.csv')
        self.history_df.to_csv(results_df_path, index = False)

        results_summary_path = os.path.join(self.folder_dir, 'results_summary')
        results_summary = [self.initial_capital, self.total_gain]
        params_path = os.path.join(self.folder_dir, 'params')
        params = [self.threshold, self.hold_till, self.sell_perc, self.stop_perc, self.start_date, self.end_date]

        with open(results_summary_path, 'wb') as fp:
            pickle.dump(results_summary, fp)
        with open(params_path, 'wb') as fp:
            pickle.dump(params, fp)

if __name__ == "__main__":
    stocks = [
    "GS"
]
    stocks = list(np.unique(stocks))
    back = backtester(stocks, LR_v1_predict, 3000, datetime(2023, 1, 1), datetime(2024, 1, 1), threshold = 0.99, sell_perc = 0.04, stop_perc = 0.03, sell_threshold = 0.02)
back.backtest()