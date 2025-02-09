import yfinance as yf
import sys
import os
import requests
import stock_utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
import pickle
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import math

class LR_training:
    def __init__(self, model_version, threshold = 0.99, start_date = None, end_date = None):
        self.model_version = model_version
        self.threshold = threshold

        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date

        stocks = [
    # Dow Jones Industrial Average (DJIA) Components
    "MMM", "AXP", "BA", "CAT", "CVX", "CSCO", "GS",
    "HD", "HON", "IBM", "INTC", "MCD", "MRK", "NKE",
    "CRM", "TRV", "VZ", "WBA", "WMT", "DIS", "XOM",

    # Top 20 S&P 500 Companies by Market Capitalization
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "NVDA", "META", "TSLA", "UNH",
    "JNJ", "JPM", "V", "PG", "MA", "PFE", "ABBV", "KO",
]
        self.stocks = list(np.unique(stocks))

        self.main_df = pd.DataFrame(columns = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg', 'target'])

        self.scaler = MinMaxScaler()
        self.lr = LogisticRegression()

        self.fetch_data()
        self.create_train_test()
        self.fit_model()
        self.confusion_matrix()
        self.save_model()

    def fetch_data(self):
        """
        fetch train and test data from yahoo finance
        """
        for stock in self.stocks:
            try:
                df = stock_utils.create_train_data(stock, n = 10)
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                self.main_df = pd.concat([self.main_df, df], axis = 0, ignore_index = True)
            except:
                pass
        print(f'{len(self.main_df)} samples were fetched from the database.')
    
    def create_train_test(self):
        """
        create train and test data
        """
        self.main_df = self.main_df.sample(frac = 1, random_state = 3). reset_index(drop = True)
        self.main_df = self.main_df.dropna(axis=1, how='all')
        self.main_df = self.main_df.dropna()
        self.main_df.columns = [col[0] if isinstance(col, tuple) else col for col in self.main_df.columns]
        self.main_df['target'] = self.main_df['target'].astype('category')
        self.main_df.columns = self.main_df.columns.astype(str)

        y = self.main_df.pop('target').to_numpy()
        y = y.reshape(y.shape[0], 1)
        x = self.scaler.fit_transform(self.main_df)

        #test train split
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, \
            test_size = 0.05, random_state = 50, shuffle = True)
        print('Created test and train data...')

    def fit_model(self):

        print('Training model...')
        self.lr.fit(self.train_x, self.train_y)
        
        #predict the test data
        self.predictions = self.lr.predict(self.test_x)
        self.score = self.lr.score(self.test_x, self.test_y)
        print(f'Logistic regression model score: {self.score}')

        #preds with threshold
        self.predictions_proba = self.lr._predict_proba_lr(self.test_x)
        self.predictions_proba_thresholded = self._threshold(self.predictions_proba, self.threshold)

    def confusion_matrix(self):
        cm = confusion_matrix(self.test_y, self.predictions)
        self.cmd = ConfusionMatrixDisplay(cm)
        
        cm_thresholded = confusion_matrix(self.test_y, self.predictions_proba_thresholded)
        self.cmd_thresholded = ConfusionMatrixDisplay(cm_thresholded)

    def _threshold(self, predictions, threshold):

        prob_thresholded = [0 if x > threshold else 1 for x in predictions[:, 0]]

        return np.array(prob_thresholded)

    def save_model(self):

        #save models
        saved_models_dir = os.path.join(os.getcwd(), 'saved_models')
        model_file = f'lr_{self.model_version}.sav'
        model_dir = os.path.join(saved_models_dir, model_file)
        pickle.dump(self.lr, open(model_dir, 'wb'))

        scaler_file = f'scaler_{self.model_version}.sav'
        scaler_dir = os.path.join(saved_models_dir, scaler_file)
        pickle.dump(self.scaler, open(scaler_dir, 'wb'))

        print(f'Saved the model and scaler in {saved_models_dir}')
        cm_path = os.path.join(os.getcwd(), 'confusion_matrices')
        
        #save cms
        plt.figure()
        self.cmd.plot()
        plt.savefig(f'{cm_path}\\cm_{self.model_version}.jpg')

        plt.figure()
        self.cmd_thresholded.plot()
        plt.savefig(f'{cm_path}\\cm_thresholded_{self.model_version}.jpg')
        print(f'Figures saved in {cm_path}')

if __name__ == "__main__":
    run_lr = LR_training('v2')