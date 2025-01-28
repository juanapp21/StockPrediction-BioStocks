import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas_datareader.data as web
import pickle
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os 
import pickle
models_path = 'saved_models/'







def do_prediction(model, ticker,interval,interval_desc):
    ## fetch data of the last 60 days only
# Get the stock quote
    date_before_60_days = datetime.datetime.now() - datetime.timedelta(days=100)
    df = yf.download(ticker, start=date_before_60_days, end=datetime.datetime.now(), interval=interval)


    # Print the first 5 rows
    print(df.head())

    # Create a new dataframe with only the 'Close column 
    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    len(dataset)
    ## load scalar 
    scaler_path = os.path.join(models_path, ticker, interval_desc,'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    scaled_data = scaler.transform(dataset)

    x = scaled_data[-60:]
    x = np.reshape(x, (1, x.shape[0], 1))
    print(x.shape)
    predictions = model.predict(x)
    predictions = scaler.inverse_transform(predictions)
    print(predictions)
    return predictions

def get_saved_models(ticker,interval_desc):
    # number of past ticks = 60 
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (60, 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.load_weights('saved_models\{}\{}\stock_prediction.h5'.format(ticker,interval_desc))
    return model


def get_current_price(ticker):
    date_before_60_days = datetime.datetime.now() - datetime.timedelta(days=100)

    df = yf.download(ticker,  start=date_before_60_days, end=datetime.datetime.now() )
    current_price = df['Close'][-1]
    return current_price

def get_prediction():
        
        appl_model_hr = get_saved_models('AAPL','hr')
        appl_model_day = get_saved_models('AAPL','day')
        appl_model_week = get_saved_models('AAPL','week')
        app_hr_predcitions   = do_prediction(appl_model_hr, 'AAPL',interval='1h',interval_desc="hr")[0][0]
        app_day_predcitions  = do_prediction(appl_model_day, 'AAPL',interval='1d',interval_desc="day")[0][0]
        app_week_predcitions = do_prediction(appl_model_week, 'AAPL',interval='1d',interval_desc="week")[0][0]
        current_aapl_price = get_current_price('AAPL')
        amzn_model_hr = get_saved_models('AMZN','hr')
        amzn_model_day = get_saved_models('AMZN','day')
        amzn_model_week = get_saved_models('AMZN','week')
        amzn_hr_predcitions   = do_prediction(appl_model_hr, 'AMZN',interval='1h',interval_desc="hr")[0][0]
        amzn_day_predcitions  = do_prediction(appl_model_day, 'AMZN',interval='1d',interval_desc="day")[0][0]
        amzn_week_predcitions = do_prediction(appl_model_week, 'AMZN',interval='1d',interval_desc="week")[0][0]
        current_amzn_price = get_current_price('AMZN')
        ## round price to 3 decimal places
        current_aapl_price = round(current_aapl_price,3)
        current_amzn_price = round(current_amzn_price,3)
        app_hr_predcitions = round(app_hr_predcitions,3)
        app_day_predcitions = round(app_day_predcitions,3)
        app_week_predcitions = round(app_week_predcitions,3)
        amzn_hr_predcitions = round(amzn_hr_predcitions,3)
        amzn_day_predcitions = round(amzn_day_predcitions,3)
        amzn_week_predcitions = round(amzn_week_predcitions,3)
        
        return [{'aapl': {'current_price': current_aapl_price, 'hr_prediction': app_hr_predcitions,
                          'day_prediction': app_day_predcitions, 'week_prediction': app_week_predcitions}},
                    {'amzn': {'current_price': current_amzn_price, 'hr_prediction': amzn_hr_predcitions,
                            'day_prediction': amzn_day_predcitions, 'week_prediction': amzn_week_predcitions}}]
