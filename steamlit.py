import streamlit as st
import pymongo
import pandas as pd
import time
import plotly.express as px

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "pjt2"

client = pymongo.MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Streamlit App Configuration
st.set_page_config(page_title="Object Detection Dashboard", layout="wide")
st.title("🚀 AI-Powered Object Detection Dashboard")

# Function to fetch counts
def get_object_counts():
    return {
        "humans": db.human_detections.count_documents({}),
        "animals": db.animal_detections.count_documents({}),
        "drones": db.drone_detections.count_documents({})
    }

# Live Counter Section
st.subheader("📊 Real-Time Detection Counts")
counts = get_object_counts()
col1, col2, col3 = st.columns(3)
col1.metric("Humans Detected", counts["humans"], delta=counts["humans"] - get_object_counts()["humans"] if st.session_state.get('last_counts', None) else 0)
col2.metric("Animals Detected", counts["animals"], delta=counts["animals"] - get_object_counts()["animals"] if st.session_state.get('last_counts', None) else 0)
col3.metric("Drones Detected", counts["drones"], delta=counts["drones"] - get_object_counts()["drones"] if st.session_state.get('last_counts', None) else 0)

if 'last_counts' not in st.session_state:
    st.session_state.last_counts = counts
else:
    st.session_state.last_counts = counts

# Function to fetch recent detections
def fetch_detections(collection_name):
    data = list(db[collection_name].find({}, {"_id": 0}).sort("timestamp", -1).limit(10))
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["timestamp", "objects_detected", "location"])

# Detection Logs Section
st.subheader("📜 Recent Detections")
tab1, tab2, tab3 = st.tabs(["Humans", "Animals", "Drones"])

with tab1:
    st.dataframe(fetch_detections("human_detections"))

with tab2:
    st.dataframe(fetch_detections("animal_detections"))

with tab3:
    st.dataframe(fetch_detections("drone_detections"))

# Bar Chart Section
st.subheader("📈 Detection Distribution")
detection_distribution = pd.DataFrame({
    "Category": ["Humans", "Animals", "Drones"],
    "Count": [counts["humans"], counts["animals"], counts["drones"]]
})
fig = px.bar(detection_distribution, x="Category", y="Count", title="Object Detection Distribution")
st.plotly_chart(fig, use_container_width=True)

# Time Series Graph
st.subheader("⏱ Detections Over Time")

def fetch_time_series_data(collection_name):
    data = list(db[collection_name].find({}, {"_id": 0}))
    df = pd.DataFrame(data)
    if not df.empty:
        # Convert timestamp to proper format
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d_%H-%M-%S", errors='coerce')
        df = df.dropna(subset=['timestamp'])  # Remove rows where timestamp conversion failed
        df.set_index('timestamp', inplace=True)
        return df.resample('1T').size().fillna(0)  # Resample every 1 minute
    return pd.Series()


human_time_series = fetch_time_series_data("human_detections")
animal_time_series = fetch_time_series_data("animal_detections")
drone_time_series = fetch_time_series_data("drone_detections")

time_series_df = pd.DataFrame({
    "Humans": human_time_series,
    "Animals": animal_time_series,
    "Drones": drone_time_series
})

# Ensure all columns are numeric (convert NaNs to 0 and set type to int)
time_series_df = time_series_df.fillna(0).astype(int)

if not time_series_df.empty:
    fig_time_series = px.line(time_series_df, title="Detections Over Time")
    st.plotly_chart(fig_time_series, use_container_width=True)


# Auto Refresh Every 5 Seconds
time.sleep(5)
st.experimental_rerun()