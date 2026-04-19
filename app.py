import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os

st.set_page_config(page_title="SUV Purchase Predictor", layout="centered")

@st.cache_data
def load_and_train_model():
    if not os.path.exists('suv_data.csv'):
        return None, None
    
    df = pd.read_csv('suv_data.csv')
    X = df.iloc[:, [2, 3]].values
    y = df.iloc[:, 4].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    return model, sc

model, sc = load_and_train_model()

def main():
    st.title("🚗 SUV Car Purchasing Prediction")
    st.write("Predict whether a customer will purchase an SUV based on Age and Salary.")
    
    if os.path.exists('suv_car.jpeg'):
        st.image('suv_car.jpeg', width=400)
    
    st.divider()

    st.sidebar.header("User Input")
    age = st.sidebar.slider("Select Age:", 18, 80, 30)
    salary = st.sidebar.slider("Select Salary:", 10000, 200000, 50000, 1000)

    if model is not None:
        X_new = np.array([[age, salary]])
        X_new_scaled = sc.transform(X_new)
        prediction = model.predict(X_new_scaled)
        
        st.subheader("Result:")
        if prediction[0] == 1:
            st.success("✅ This person will buy the SUV.")
        else:
            st.error("❌ This person will not buy the SUV.")
    else:
        st.error("File 'suv_data.csv' not found!")

if __name__ == '__main__':
    main()