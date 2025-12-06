import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

st.title("Housing Price Prediction App")
st.write("This app uses to predict house prices.")



# Load Model & Pipeline

if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
    st.error(" model.pkl or pipeline.pkl missing. Please train your model first.")
    st.stop()

model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)
st.success(" Model & Pipeline Loaded Successfully!")



# CSV Inference Options

st.subheader("Inference on CSV (input.csv)")

if os.path.exists("input.csv"):
    df = pd.read_csv("input.csv")
    st.write("Input CSV Preview:")
    st.dataframe(df)

    if st.button("Run Inference on input.csv"):
        transformed = pipeline.transform(df)
        preds = model.predict(transformed)

        df_out = df.copy()
        df_out["median_house_value"] = preds

        df_out.to_csv("output.csv", index=False)
        st.success("output.csv generated successfully!")

        st.download_button(
            "⬇ Download output.csv",
            df_out.to_csv(index=False),
            "output.csv",
            "text/csv"
        )
else:
    st.info("input.csv not found. Upload or generate one.")



# Manual Prediction Form

st.subheader(" Predict House Price")

col1, col2 = st.columns(2)

longitude = col1.number_input("Longitude", -125.0, -110.0, step=0.01)
latitude = col2.number_input("Latitude", 32.0, 45.0, step=0.01)

housing_median_age = col1.number_input("Housing Median Age", 0, 100)
total_rooms = col2.number_input("Total Rooms", 0, 50000)
total_bedrooms = col1.number_input("Total Bedrooms", 0, 10000)
population = col2.number_input("Population", 0, 50000)
households = col1.number_input("Households", 0, 50000)
median_income = col2.number_input("Median Income", 0.0, 20.0, step=0.1)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

if st.button("Predict Price"):
    row = pd.DataFrame([[
        longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
        population, households, median_income, ocean_proximity
    ]], columns=[
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
        "ocean_proximity"
    ])

    transformed = pipeline.transform(row)
    prediction = model.predict(transformed)[0]

    st.success(f" Predicted House Price: **${prediction:,.2f}**")
