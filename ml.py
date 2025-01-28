import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.utils import shuffle

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(os.getcwd(), 'saved_models')

# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

def prepare_dataset(ticker, from_date, end_date, save_scaler=True):
    # Download stock data with multiple features
    dfs = []
    data = yf.download(ticker, start=from_date, end=end_date)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Selecting OHLCV columns
        ## append 

    # Append the prepared DataFrame to the list
    dfs.append(data)
    full_data = pd.concat(dfs)
    ## shuffle the data
    #full_data = full_data.sample(frac=1).reset_index(drop=True)
    # Normalize the dat aset
    features_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = features_scaler.fit_transform(full_data)
    
    target = full_data[['Close']]  # Target variable

    # Normalize the target variable separately
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(target)

    # Save the scalers for later use
    if save_scaler:
        features_saving_path = MODELS_PATH +  f'/{ticker}_features_scaler.pkl'
        target_saving_path = MODELS_PATH +  f'/{ticker}_target_scaler.pkl'
        print('features_saving_path:', features_saving_path)
        with open(features_saving_path, 'wb') as f:
            pickle.dump(features_scaler, f)

        with open(target_saving_path, 'wb') as f:
            pickle.dump(target_scaler, f)


    return scaled_data, scaled_target

def create_dataset_multifeature(data, scaled_target, time_step=1, prediction_step=10):
    X, Y = [], []
    for i in range(len(data) - time_step - prediction_step - 1):
        X.append(data[i:(i + time_step), :])
        # Y is now the 'Close' price at 'prediction_step' days ahead
        Y.append(scaled_target[i + time_step + prediction_step - 1, 0])  # 0 index, assuming single column in scaled_target
    return shuffle(np.array(X), np.array(Y), random_state=42)



# Example of how to use these functions:

def prepare_data_for_future_prediction(ticker, time_step, pred_step):
    # Calculate start date with buffer for weekends/holidays and prediction step
    today = datetime.now()
    buffer_days = 2  # Adjust based on typical non-trading days in the market
    start_date_buffer = (time_step + pred_step) * buffer_days
    start_date = (today - timedelta(days=start_date_buffer)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    
    # Download the data
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        print(data.tail())
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(len(data))
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    
    # Ensure the scaler used here is the same one used for model training
    try:
        scaler_path = MODELS_PATH +  f'/{ticker}_features_scaler.pkl'
        scaler = pickle.load(open(scaler_path, 'rb'))
    except Exception as e:
        print('path used : ' + scaler_path) 
        print(f"Error loading scaler xxx: {e}")
        return None
    
    # Scaling the data
    data_scaled = scaler.transform(data)
    X_pred = np.array([data_scaled])
    return X_pred   

def do_live_prediction(ticker,time_step, pred_step,model_path):
    X_pred = prepare_data_for_future_prediction(ticker, time_step, pred_step)
    target_path =  MODELS_PATH +  f'/{ticker}_target_scaler.pkl'
    print('target path: *********** ', target_path)
    model = get_model(model_path, True )
    
    print('scaler path : ')
    ## load scaler 
    target_scaler = pickle.load(open(target_path, 'rb'))
    print('target scaler:', target_scaler)
    ## load model
    prediction = model.predict(X_pred)
    transformed_pred = target_scaler.inverse_transform(prediction)
    return transformed_pred



def get_model(model_path='', load_weights=False):
        model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(100, 5)),  # 100 time steps, 5 features
        LSTM(128),
        Dense(1)
    ])
        if load_weights:
            model.load_weights(model_path)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        return model



# Assuming X and y datasets as prepared in the earlier step
def train_lstm(model,ticker, X, y, epochs=50, save_path='lstm_model.keras'):
# Reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 5))  # 5 for OHLCV features

    # Define the LSTM model
    # Fit the model
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=2, callbacks=[early_stop])
    model.save(save_path)
    return model


# Assuming X and y datasets as prepared in the earlier step
def train_lstm(model, X, y, epochs=50, save_path='lstm_model.keras'):
# Reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 5))  # 5 for OHLCV features

    # Define the LSTM model
    # Fit the model
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.15, verbose=2, callbacks=[early_stop])
    model.save(save_path)
    return model

def train_models():
    symbols = ['NVAX']
    current_date = datetime.now().strftime('%Y-%m-%d')
    prev_5_yrs_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')

    time_step = 100
    pred_step = 1


    for ticker in symbols: 
        save_path =  MODELS_PATH +  f'/day_{ticker}_lstm_model.keras'
        features, targets = prepare_dataset(ticker, prev_5_yrs_date, current_date)
        X, y = create_dataset_multifeature(features, targets, time_step, pred_step)
        model = get_model()
        model = train_lstm(model, X, y, 30,save_path)


    time_step = 100
    pred_step = 30
    for ticker in symbols: 
        save_path =  MODELS_PATH +  f'/month_{ticker}_lstm_model.keras'

        features, targets = prepare_dataset(ticker, prev_5_yrs_date, current_date)
        X, y = create_dataset_multifeature(features, targets, time_step, pred_step)
        model = get_model()
        model = train_lstm(model, X, y, 30,save_path)
    
    time_step = 100
    pred_step = 7
    for ticker in symbols: 
        save_path =   MODELS_PATH +  f'/week_{ticker}_lstm_model.keras'

        features, targets = prepare_dataset(ticker, prev_5_yrs_date, current_date)
        X, y = create_dataset_multifeature(features, targets, time_step, pred_step)
        model = get_model()
        model = train_lstm(model, X, y, 30,save_path)

# Usage example
