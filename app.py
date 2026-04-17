import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.cluster import KMeans
import warnings
import os

# ==================================================
# Suppress warnings
# ==================================================
warnings.filterwarnings("ignore")

# ==================================================
# Page Config
# ==================================================
st.set_page_config(page_title="House Price Predictor", layout="wide")

# ==================================================
# Correct Base Path
# ==================================================
BASE_PATH = r"D:\house price prediction app\backend"

MODEL_PATH = os.path.join(BASE_PATH, "final_voting_regressor_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(BASE_PATH, "model_feature_names.joblib")
TRAIN_DATA_PATH = os.path.join(BASE_PATH, "train house.csv")
TEST_DATA_PATH = os.path.join(BASE_PATH, "test house.csv")

# ==================================================
# Safe File Check
# ==================================================
def check_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


# ==================================================
# Load Model + Feature Names
# ==================================================
@st.cache_resource
def load_model_and_features():
    try:
        # Load Model
        model = joblib.load(MODEL_PATH)

        # -----------------------------
        # FIX JSON ERROR HERE
        # -----------------------------
        if not check_file(FEATURE_COLUMNS_PATH):
            st.error("Feature column file missing or empty.")
            st.stop()

        # Try Joblib first
        try:
            feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

        except:
            # Try JSON second
            with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()

                if content == "":
                    st.error("Feature column file is empty.")
                    st.stop()

                feature_columns = json.loads(content)

        return model, feature_columns

    except Exception as e:
        st.error(f"Loading Error: {e}")
        st.stop()


final_model, model_feature_columns = load_model_and_features()

# ==================================================
# Load Train/Test Data
# ==================================================
@st.cache_data
def load_auxiliary_data():
    try:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)

        # Remove duplicates
        train_df.drop_duplicates(inplace=True)

        # Remove Outliers
        Q1 = train_df["TARGET(PRICE_IN_LACS)"].quantile(0.25)
        Q3 = train_df["TARGET(PRICE_IN_LACS)"].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 3 * IQR

        train_df = train_df[
            train_df["TARGET(PRICE_IN_LACS)"] <= upper
        ]

        # Log Target
        train_df["Price_log"] = np.log1p(
            train_df["TARGET(PRICE_IN_LACS)"]
        )

        # Address Encoding
        address_mapping = train_df.groupby("ADDRESS")[
            "Price_log"
        ].mean()

        global_mean = train_df["Price_log"].mean()

        # Fill Missing Coordinates
        test_df["LONGITUDE"].fillna(
            train_df["LONGITUDE"].mean(),
            inplace=True
        )

        test_df["LATITUDE"].fillna(
            train_df["LATITUDE"].mean(),
            inplace=True
        )

        # KMeans Cluster
        coords = pd.concat([
            train_df[["LONGITUDE", "LATITUDE"]],
            test_df[["LONGITUDE", "LATITUDE"]]
        ])

        kmeans = KMeans(
            n_clusters=5,
            random_state=42,
            n_init=10
        )

        kmeans.fit(coords)

        addresses = sorted(
            address_mapping.index.tolist()
        )

        return (
            address_mapping.to_dict(),
            global_mean,
            kmeans,
            addresses
        )

    except Exception as e:
        st.error(f"CSV Loading Error: {e}")
        st.stop()


address_mapping, global_mean_price_log, kmeans_model, unique_addresses = load_auxiliary_data()

# ==================================================
# UI
# ==================================================
st.title("🏡 House Price Prediction App")
st.write("Enter house details to predict house price")

st.sidebar.header("Property Details")

posted_by = st.sidebar.selectbox(
    "Posted By",
    ["Owner", "Dealer", "Builder"]
)

bhk_or_rk = st.sidebar.selectbox(
    "Type",
    ["BHK", "RK"]
)

under_construction = st.sidebar.radio(
    "Under Construction",
    ["No", "Yes"]
) == "Yes"

rera = st.sidebar.radio(
    "RERA Approved",
    ["No", "Yes"]
) == "Yes"

ready_to_move = st.sidebar.radio(
    "Ready To Move",
    ["Yes", "No"]
) == "Yes"

resale = st.sidebar.radio(
    "Resale",
    ["No", "Yes"]
) == "Yes"

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

# ==================================================
# Prediction
# ==================================================
if st.sidebar.button("Predict Price"):

    try:
        input_data = {
            "UNDER_CONSTRUCTION": int(under_construction),
            "RERA": int(rera),
            "BHK_NO.": bhk_no,
            "SQUARE_FT": square_ft,
            "READY_TO_MOVE": int(ready_to_move),
            "RESALE": int(resale),
            "LONGITUDE": longitude,
            "LATITUDE": latitude,
            "POSTED_BY_Dealer": 0,
            "POSTED_BY_Owner": 0,
            "BHK_OR_RK_RK": 0
        }

        if posted_by == "Dealer":
            input_data["POSTED_BY_Dealer"] = 1

        elif posted_by == "Owner":
            input_data["POSTED_BY_Owner"] = 1

        if bhk_or_rk == "RK":
            input_data["BHK_OR_RK_RK"] = 1

        # Cluster
        coords = pd.DataFrame(
            [[longitude, latitude]],
            columns=["LONGITUDE", "LATITUDE"]
        )

        input_data["Location_Cluster"] = \
            kmeans_model.predict(coords)[0]

        # Address Encoding
        input_data["ADDRESS_Target_Encoded"] = \
            address_mapping.get(
                address_input,
                global_mean_price_log
            )

        # DataFrame
        input_df = pd.DataFrame([input_data])

        # Reorder Features
        input_df = input_df.reindex(
            columns=model_feature_columns,
            fill_value=0
        )

        # Predict
        pred_log = final_model.predict(input_df)[0]

        pred_price = np.expm1(pred_log)

        st.success(
            f"🏠 Estimated Price: ₹ {pred_price:,.2f} Lacs"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
