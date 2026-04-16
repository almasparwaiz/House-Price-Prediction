import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.cluster import KMeans
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 0. Constants and Initial Setup ---
# ✅ FIXED PATHS using raw strings (r"")
BASE_PATH = r"D:\house price prediction app\backend"

MODEL_PATH = os.path.join(BASE_PATH, "final_house_price_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(BASE_PATH, "model_feature_columns.json")
TRAIN_DATA_PATH = os.path.join(BASE_PATH, "train house.csv")
TEST_DATA_PATH = os.path.join(BASE_PATH, "test house.csv")

# --- 1. Load Model and Feature Columns ---
@st.cache_resource
def load_model_and_features():
    model = joblib.load(MODEL_PATH)

    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_columns = json.load(f)

    return model, feature_columns

final_model, model_feature_columns = load_model_and_features()

# --- 2. Load Auxiliary Data ---
@st.cache_data
def load_and_preprocess_auxiliary_data():

    train_df_original = pd.read_csv(TRAIN_DATA_PATH)
    test_df_original = pd.read_csv(TEST_DATA_PATH)

    train_df_original.drop_duplicates(inplace=True)

    Q1 = train_df_original['TARGET(PRICE_IN_LACS)'].quantile(0.25)
    Q3 = train_df_original['TARGET(PRICE_IN_LACS)'].quantile(0.75)
    IQR = Q3 - Q1

    upper_bound = Q3 + 3 * IQR

    train_df_original = train_df_original[
        train_df_original['TARGET(PRICE_IN_LACS)'] <= upper_bound
    ]

    train_df_original['Price_log'] = np.log1p(
        train_df_original['TARGET(PRICE_IN_LACS)']
    )

    address_mapping_series = train_df_original.groupby(
        'ADDRESS'
    )['Price_log'].mean()

    global_mean_price_log = train_df_original['Price_log'].mean()

    test_df_original['LONGITUDE'].fillna(
        train_df_original['LONGITUDE'].mean(), inplace=True
    )

    test_df_original['LATITUDE'].fillna(
        train_df_original['LATITUDE'].mean(), inplace=True
    )

    combined_coords = pd.concat([
        train_df_original[['LONGITUDE', 'LATITUDE']],
        test_df_original[['LONGITUDE', 'LATITUDE']]
    ])

    kmeans_model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    kmeans_model.fit(combined_coords)

    unique_addresses = sorted(address_mapping_series.index.tolist())

    return (
        address_mapping_series.to_dict(),
        global_mean_price_log,
        kmeans_model,
        unique_addresses
    )

address_mapping, global_mean_price_log, kmeans_model, unique_addresses = load_and_preprocess_auxiliary_data()

# --- 3. Streamlit UI ---
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏡 House Price Prediction App")
st.write("Enter house details to predict price")

st.sidebar.header("Enter Property Details")

posted_by = st.sidebar.selectbox(
    "Posted By",
    ['Owner', 'Dealer', 'Builder']
)

bhk_or_rk = st.sidebar.selectbox(
    "Type",
    ['BHK', 'RK']
)

under_construction = st.sidebar.radio(
    "Under Construction",
    ['No', 'Yes']
) == 'Yes'

rera = st.sidebar.radio(
    "RERA Approved",
    ['No', 'Yes']
) == 'Yes'

ready_to_move = st.sidebar.radio(
    "Ready To Move",
    ['Yes', 'No']
) == 'Yes'

resale = st.sidebar.radio(
    "Resale",
    ['No', 'Yes']
) == 'Yes'

bhk_no = st.sidebar.number_input(
    "BHK Number",
    1, 10, 2
)

square_ft = st.sidebar.number_input(
    "Square Feet",
    100.0, 50000.0, 1200.0
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=77.0
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=12.0
)

address_input = st.sidebar.selectbox(
    "Address",
    unique_addresses
)

# --- Prediction ---
if st.sidebar.button("Predict Price"):

    input_data = {
        'UNDER_CONSTRUCTION': int(under_construction),
        'RERA': int(rera),
        'BHK_NO.': bhk_no,
        'SQUARE_FT': square_ft,
        'READY_TO_MOVE': int(ready_to_move),
        'RESALE': int(resale),
        'LONGITUDE': longitude,
        'LATITUDE': latitude,
        'POSTED_BY_Dealer': 0,
        'POSTED_BY_Owner': 0,
        'BHK_OR_RK_RK': 0
    }

    if posted_by == "Dealer":
        input_data['POSTED_BY_Dealer'] = 1

    elif posted_by == "Owner":
        input_data['POSTED_BY_Owner'] = 1

    if bhk_or_rk == "RK":
        input_data['BHK_OR_RK_RK'] = 1

    location_coords = pd.DataFrame(
        [[longitude, latitude]],
        columns=['LONGITUDE', 'LATITUDE']
    )

    input_data['Location_Cluster'] = kmeans_model.predict(location_coords)[0]

    input_data['ADDRESS_Target_Encoded'] = address_mapping.get(
        address_input,
        global_mean_price_log
    )

    input_df = pd.DataFrame([input_data])

    input_df = input_df[model_feature_columns]

    try:
        predicted_log = final_model.predict(input_df)[0]

        predicted_price = np.expm1(predicted_log)

        st.success(
            f"Estimated Price: ₹ {predicted_price:,.2f} Lacs"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")