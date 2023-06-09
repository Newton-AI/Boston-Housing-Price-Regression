import streamlit as st
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

st.set_page_config('Boston Housing Price Predictor', ':house:')
st.title('Boston Housing Price Predictor :house:')
tab1, tab2 = st.tabs(['Predictions', 'About'])

folder_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(folder_path, 'model.pkl'))
ss = StandardScaler()

def predict():
    features = np.array([[rm, lstat, dis, crim, age, tax, nox, ptratio]])
    scaled_features = ss.fit_transform(features.T)

    pred = model.predict(scaled_features.T)
    res = round(pred[0] * 1000)

    st.info(f':dollar: AI predicts the housing price to be: **$ {res}**')

with tab1:
    rm = st.number_input('Average number of rooms per dwelling', step=0.1)
    lstat = st.number_input('Lower status of the population (percent)', min_value=0.0, max_value=100.0, step=0.1)
    dis = st.number_input('Weighted mean of distances to five Boston employment centres', step=0.1)
    crim = st.number_input('Per capita crime rate by town', step=0.1)
    age = st.number_input('Proportion of owner-occupied units built prior to 1940.', step=0.1)
    tax = st.number_input('Full-value property-tax rate per \$10,000.', step=1)
    nox = st.number_input('Nitrogen oxides concentration (parts per 10 million).', min_value=0.0, max_value=1.0, step=0.01)
    ptratio = st.number_input('Pupil-teacher ratio by town.', step=0.1)

    st.button('Predict', on_click=predict)

with tab2:
    st.markdown('This model is trained using **:blue[Random Forest Regressor]** Model.')

    col1, col2, col3 = st.columns(3)
    col1.metric(label='R2 Score', value='89%')
    col2.metric(label='RMSE', value=2.62)
    col3.metric(label='MAE', value=1.93)

    st.image(os.path.join(folder_path, 'model report.png'), caption='Compare labels and predictions for testing data')
