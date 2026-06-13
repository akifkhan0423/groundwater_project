import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Groundwater Predictor",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#1E88E5;'>
    Groundwater Depletion Risk Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("---")

data = pd.read_csv("groundwater.csv")

label = LabelEncoder()

data["Risk"] = label.fit_transform(data["Risk"])

data["WaterStress"] = (
    data["Temperature"] / data["Rainfall"]
)

X = data[
    [
        "Rainfall",
        "Temperature",
        "SoilMoisture",
        "WaterStress"
    ]
]

y = data["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=1
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

log_model = LogisticRegression()

log_model.fit(X_train, y_train)

lin_model = LinearRegression()

lin_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")

st.success( 
    f"Accuracy : {round(accuracy * 100, 2)}%"
)

st.write("---")

st.subheader("Enter Environmental Details")

rainfall = st.slider(
    "Rainfall (mm)",
    100,
    1000,
    600
)

temperature = st.slider(
    "Temperature (°C)",
    20,
    45,
    30
)

soil = st.slider(
    "Soil Moisture",
    0.1,
    1.0,
    0.5
)

if st.button("Predict Risk"):

    water_stress = temperature / rainfall

    input_data = [[
        rainfall,
        temperature,
        soil,
        water_stress
    ]]

    input_data = scaler.transform(input_data)

    prediction = log_model.predict(input_data)

    risk = label.inverse_transform(prediction)

    linear_value = lin_model.predict(input_data)

    st.write("---")

    st.subheader("Prediction Result")

    if risk[0] == "High":
        st.error("High Groundwater Risk")

    elif risk[0] == "Medium":
        st.warning("Medium Groundwater Risk")

    else:
        st.success("Low Groundwater Risk")

    st.info(
        f"Linear Regression Value : {round(linear_value[0], 2)}"
    )

st.write("---")

st.subheader("Rainfall vs Groundwater Depth")

fig, ax = plt.subplots()

ax.scatter(
    data["Rainfall"],
    data["GroundwaterDepth"]
)

ax.set_xlabel("Rainfall")

ax.set_ylabel("Groundwater Depth")

st.pyplot(fig)

st.write("---")

st.caption(
    "Mini Project using Machine Learning and Streamlit"
)