import streamlit as st
import numpy as np
import pickle
import os

st.markdown("""
<style>

/* Full page background */
html, body, .stApp {
    min-height: 100vh;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Center YouTube logo */
.stApp {
    position: relative;
    background-image:
        linear-gradient(
            rgba(15,32,39,0.94),
            rgba(32,58,67,0.94),
            rgba(44,83,100,0.94)
        ),
        url("https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg");
    background-repeat: no-repeat;
    background-position: center;
    background-size: 260px;
}

/* Right-side YouTube logo */
.stApp::after {
    content: "";
    position: fixed;
    top: 50%;
    right: 4%;
    transform: translateY(-50%);
    width: 220px;
    height: 220px;
    background-image: url("https://upload.wikimedia.org/wikipedia/commons/0/09/YouTube_full-color_icon_%282017%29.svg");
    background-repeat: no-repeat;
    background-size: contain;
    opacity: 0.08;
    pointer-events: none;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22 !important;
}

/* Text */
h1, h2, h3, p, label, span, div {
    color: #e6e6e6 !important;
}

/* Inputs */
input, textarea, select {
    background-color: #0e1117 !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #30363d !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    border-radius: 12px;
}

/* Dropdown menu */
div[role="listbox"] {
    background-color: #161b22 !important;
}
div[role="option"] {
    color: white !important;
}
div[role="option"]:hover {
    background-color: #3fb950 !important;
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Whole selectbox control */
div[data-baseweb="select"] > div {
    background-color: #0e1117 !important;
    color: white !important;
    border-radius: 10px;
}

/* Text inside selectbox */
div[data-baseweb="select"] span {
    color: white !important;
}

/* Dropdown arrow */
div[data-baseweb="select"] svg {
    fill: white !important;
}

/* Dropdown menu */
ul[role="listbox"] {
    background-color: #161b22 !important;
}

/* Dropdown options */
li[role="option"] {
    background-color: #161b22 !important;
    color: white !important;
}

/* Hover */
li[role="option"]:hover {
    background-color: #238636 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)






# ------------------ Load model & scaler ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "lasso_pipeline.pkl"), "rb") as f:
    model = pickle.load(f)


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="YouTube Ad Revenue Predictor", layout="centered")

st.title("📊 YouTube Ad Revenue Prediction")
st.write("Predict YouTube ad revenue using performance, engagement, and metadata.")

st.sidebar.header("📌 Video Details")
st.markdown("### 📈 Engagement Metrics")
 

# ------------------ Numeric Inputs ------------------
views = st.sidebar.number_input("Views", min_value=0, value=10000)
likes = st.sidebar.number_input("Likes", min_value=0, value=500)
comments = st.sidebar.number_input("Comments", min_value=0, value=50)

watch_time_minutes = st.sidebar.number_input(
    "Total Watch Time (minutes)", min_value=0.0, value=2000.0
)
video_length_minutes = st.sidebar.number_input(
    "Video Length (minutes)", min_value=0.1, value=10.0
)
subscribers = st.sidebar.number_input("Subscribers", min_value=0, value=10000)

day = st.sidebar.number_input("Upload Day", 1, 31, 15)
month = st.sidebar.number_input("Upload Month", 1, 12, 6)
year = st.sidebar.number_input("Upload Year", 2020, 2030, 2024)

# ------------------ Categorical Inputs ------------------
category = st.sidebar.selectbox(
    "Category",
    ["Entertainment", "Gaming", "Lifestyle", "Music", "Tech"]
)

device = st.sidebar.selectbox(
    "Device",
    ["Mobile", "TV", "Tablet"]
)

country = st.sidebar.selectbox(
    "Country",
    ["CA", "DE", "IN", "UK", "US"]
)
currency = st.sidebar.radio(
    "💱 Select Currency",
    ["USD ($)", "INR (₹)"],
    horizontal=True
)

# ------------------ One-Hot Encoding ------------------
category_map = {
    "Entertainment": [1, 0, 0, 0, 0],
    "Gaming": [0, 1, 0, 0, 0],
    "Lifestyle": [0, 0, 1, 0, 0],
    "Music": [0, 0, 0, 1, 0],
    "Tech": [0, 0, 0, 0, 1]
}

device_map = {
    "Mobile": [1, 0, 0],
    "TV": [0, 1, 0],
    "Tablet": [0, 0, 1]
}

country_map = {
    "CA": [1, 0, 0, 0, 0],
    "DE": [0, 1, 0, 0, 0],
    "IN": [0, 0, 1, 0, 0],
    "UK": [0, 0, 0, 1, 0],
    "US": [0, 0, 0, 0, 1]
}

category_features = category_map[category]
device_features = device_map[device]
country_features = country_map[country]

# ------------------ Feature Engineering ------------------
engagement_rate = (likes + comments) / (views + 1)
like_view_ratio = likes / (views + 1)
comment_view_ratio = comments / (views + 1)
avg_watch_time_per_view = watch_time_minutes / (views + 1)
watch_time_ratio = avg_watch_time_per_view / (video_length_minutes + 0.01)
views_per_subscriber = views / (subscribers + 1)
engagement_weighted_watchtime = engagement_rate * watch_time_minutes

# ------------------ Feature Vector (ORDER MATTERS) ------------------
features = np.array([[
    views,
    likes,
    comments,
    watch_time_minutes,
    video_length_minutes,
    subscribers,
    day,
    month,
    year,

    *category_features,
    *device_features,
    *country_features,

    engagement_rate,
    like_view_ratio,
    comment_view_ratio,
    avg_watch_time_per_view,
    watch_time_ratio,
    views_per_subscriber,
    engagement_weighted_watchtime
]])

# ------------------ Prediction ------------------
 
USD_TO_INR = 83.0  # approximate, interview-
     # ------------------ Prediction ------------------
if st.button("💰 Predict Ad Revenue"):
    prediction_usd = model.predict(features)[0]

    if currency == "INR (₹)":
        prediction_value = prediction_usd * USD_TO_INR
        symbol = "₹"
    else:
        prediction_value = prediction_usd
        symbol = "$"

    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.1);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        font-size: 22px;
    ">
        💰 <strong>Estimated Ad Revenue</strong><br>
        <span style="font-size: 32px; color: #00e676;">
            {symbol}{prediction_value:,.2f}
        </span>
    </div>
    """, unsafe_allow_html=True)


      