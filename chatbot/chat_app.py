import streamlit as st
from extract_location_time import extract_location_time
from build_features import build_features
import requests
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ✅ Check if environment variables loaded
if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("RENDER_API_URL"):
    st.error("❌ API keys not loaded properly. Please check your .env file!")
    st.stop()

# Load historical crime stats dataset
crime_stats_data = pd.read_csv("data/crime_stats.csv")

# Page setup
st.set_page_config(
    page_title="🚨 Baltimore Crime Chatbot",
    page_icon="🚓",
)

st.title("🚓 Baltimore Crime Chatbot")

# Initial bot greeting
if "greeted" not in st.session_state:
    st.chat_message("assistant").write("""
👋 Welcome to **Baltimore Crime Chatbot**!
🔍 Powered by **XGBoost** machine learning model.
🎯 Model Accuracy: **75.99%** on historical Baltimore crime data.
📊 I can predict likely crime types based on **location and time** you ask!
💡 Use me to stay informed and stay safe!
""")
    st.session_state.greeted = True

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Enter your question:")

# Helper to parse time string like "2AM", "10PM" to integer hour

def parse_time_string(time_str):
    try:
        time_str = time_str.strip().upper()
        if "AM" in time_str:
            hour = int(time_str.replace("AM", "").strip())
            return 0 if hour == 12 else hour
        elif "PM" in time_str:
            hour = int(time_str.replace("PM", "").strip())
            return hour if hour == 12 else hour + 12
        else:
            return int(time_str)  # Assume it's already an int hour
    except:
        return None

# Function to get historical crime insights
def get_crime_insights(location):
    filtered = crime_stats_data[crime_stats_data["Neighborhood"].str.lower() == location.lower()]
    if filtered.empty:
        return None
    crime_counts = filtered["Description"].value_counts().head(5)
    return crime_counts

# Function to show pie chart
def show_crime_pie(crime_counts):
    fig, ax = plt.subplots()
    crime_counts.plot.pie(autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

# Safety recommendations by predicted crime type
def safety_tip(predicted_crime):
    tips = {
        "Robbery": "Stay in well-lit areas. Avoid walking alone late at night.",
        "Assault": "Be aware of your surroundings. Stay in public areas.",
        "Theft": "Keep valuables hidden and stay alert in crowded places.",
        "Burglary": "Secure doors and windows. Report suspicious activities.",
    }
    return tips.get(predicted_crime, "Stay aware and cautious wherever you go!")

# Handle special FAQ-type questions
if user_input:
    lower_input = user_input.lower()
    if "model" in lower_input:
        st.chat_message("assistant").write("🧠 I use an XGBoost Classifier trained on Baltimore crime, weather, and demographic data.")
    elif "accuracy" in lower_input:
        st.chat_message("assistant").write("🎯 My model achieved a prediction accuracy of 75.99% on test data.")
    elif "what can you do" in lower_input or "help" in lower_input:
        st.chat_message("assistant").write("💬 I can predict likely crime types based on the location and time you ask about. I can also show you historical crime patterns and offer safety tips.")
    elif "thank" in lower_input:
        st.chat_message("assistant").write("🤖 Thank you for using Baltimore Crime Chatbot. Stay safe. 🚓🛡️")

# Main bot logic
if st.button("Submit"):
    if user_input:
        # Save user message
        st.chat_message("user").write(user_input)

        # 1️⃣ Extract location and time
        extracted = extract_location_time(user_input)
        location = extracted.get("location", "unknown")
        time = extracted.get("time", "unknown")

        st.chat_message("assistant").write(f"📍 **Location detected:** {location}\n🕑 **Time detected:** {time}\n(Analyzing crime safety...)")

        # 2️⃣ Build features
        try:
            hour = parse_time_string(time)
            if hour is None:
                st.chat_message("assistant").write("❗ Unable to understand the time format.")
            else:
                features = build_features(location, hour)
                if features is None:
                    st.chat_message("assistant").write("❗ Sorry, no historical data found for this location and time.")
                else:
                    # Convert features to pure Python types
                    features = json.loads(json.dumps(features, default=lambda x: x.item() if hasattr(x, 'item') else x))

                    # 3️⃣ Call Render API
                    api_url = os.getenv("RENDER_API_URL")
                    headers = {"Content-Type": "application/json"}
                    response = requests.post(api_url, json=features, headers=headers)
                    prediction = response.json().get("prediction", "Unknown")

                    # Save in session state
                    st.session_state['location'] = location
                    st.session_state['prediction'] = prediction

                    # 4️⃣ Show prediction
                    st.chat_message("assistant").write(f"🔮 Based on our analysis, predicted crime type: **{prediction}**.")

                    # 5️⃣ Give safety tip
                    st.chat_message("assistant").write(f"🛡️ **Safety Tip:** {safety_tip(prediction)}")

                    # 6️⃣ Red Alert for dangerous crimes
                    if prediction in ["Robbery", "Assault", "Homicide"]:
                        st.chat_message("assistant").write("🔴 **ALERT:** High danger detected. Please exercise extreme caution!")
        except Exception as e:
            st.chat_message("assistant").write(f"❗ Error processing your request: {e}")

# 7️⃣ Show Crime Stats if button clicked after submit
if "location" in st.session_state and st.button(f"📊 See Crime Stats for {st.session_state['location']}"):
    crime_counts = get_crime_insights(st.session_state['location'])
    if crime_counts is not None:
        show_crime_pie(crime_counts)
        st.write("### Historical Crime Distribution")
        for crime, count in crime_counts.items():
            st.write(f"- {crime}: {count} cases")
    else:
        st.write("❗ No historical crime data available for this location.")