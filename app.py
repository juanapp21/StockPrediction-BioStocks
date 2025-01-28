import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# Function to train the model
import streamlit as st
import yfinance as yf
import ml 
import os 
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(os.getcwd(), 'saved_models')
print('model path:', MODELS_PATH)
print(MODELS_PATH)
# Function to simulate training a model
def train_model():
    st.write("Model training...")
    ml.train_models()
    st.write("Model training Completed !")
# Function to get and display the current price of selected stock
def display_current_stock_price(selected_stock):
    ticker_symbol = selected_stock
    ticker_data = yf.Ticker(ticker_symbol)
    ticker_df = ticker_data.history(period='5d')
    print(ticker_df)
    current_price = ticker_df['Close'].iloc[-1]
    st.write(f"Current Price of {selected_stock}: $", round(current_price, 2))

# Function to simulate price prediction for next day and next month for the selected stock
def predict_prices(selected_stock):
    if 'selected_stock' in ["RIVN", "ATNM", "IZO.CN", "CYBCF"]:
        return
    time_step = 100
    day_pred_step = 1
    month_pred_step = 30
    #get current working directory 

    #day_model_path = f'saved_models/day_{selected_stock}_lstm_model.keras'
    day_model_path = MODELS_PATH +  f'/day_{selected_stock}_lstm_model.keras'
    print('day_model_path:', day_model_path)
    day_pred = ml.do_live_prediction(selected_stock, time_step, day_pred_step,day_model_path)
    
    #month_model_path = f'saved_models/month_{selected_stock}_lstm_model.keras'
    month_model_path = MODELS_PATH +   f'/month_{selected_stock}_lstm_model.keras'

    month_pred = ml.do_live_prediction(selected_stock, time_step, month_pred_step,month_model_path)

    # These are placeholder functions. Replace with actual prediction logic.
    st.write(f"Predicted Price for next day for: {day_pred}")
    st.write(f"Predicted Price for next month for: {month_pred}")

# Streamlit App Design
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #000000;
    color:#ffffff;
}
</style>""", unsafe_allow_html=True)

st.title("Stock Prediction App")

# Dropdown for stock selection
stock_options = ['RIVN', 'ATNM', 'IZO.CN', 'CYBCF']

# Section 1: Model Training
st.header("Section 1: Training")
if st.button("Train Model"):
    train_model()

# Adding more vertical space between the two sections
st.write("\n\n")  # Adds vertical space
st.write ("---")  # Adds a horizontal line
selected_stock = st.selectbox('Select a stock for prediction:', stock_options)
st.write("\n\n")  # Adds vertical space
st.write("\n\n")  # Adds vertical space

# Section 2: Stock Prices and Predictions
st.header(f"Section 2: {selected_stock} Stock Prices and Predictions")
display_current_stock_price(selected_stock)

if st.button("Predict Next Day and Month"):
    predict_prices(selected_stock)
     
if __name__ == "__main__":
       
    # Run the app
    print('starting streamlit app...')
    
    
