# 🚓 Baltimore Crime Prediction & Safety Chatbot

## 📚 Overview
This project combines a machine learning model and a Streamlit chatbot to help users assess crime risks in Baltimore based on location and time.

- 🧠 Predicts likely crime types (XGBoost classifier)
- 🛡️ Provides safety tips and alerts
- 📊 Displays historical crime insights
- 🤖 Natural chatbot interaction for public safety

---

## 🏗️ Project Structure
```
crime-prediction-api/
├── app.py                # Backend API (Flask)
├── model/                 # Trained XGBoost model
├── utils/                 # Feature engineering scripts
├── data/                  # Processed datasets
├── chatbot/               # Streamlit Chatbot Frontend
│   ├── chat_app.py
│   ├── build_features.py
│   ├── extract_location_time.py
│   └── data/crime_stats.csv
├── requirements.txt
└── README.md
```

---

## 🔍 Key Features
- 📍 Extracts **location** and **time** from user queries
- 🔮 Predicts crime types using an **XGBoost model (~76% accuracy)**
- 🗺️ Shows crime distribution as **pie charts**
- 🛡️ Offers **contextual safety recommendations**
- 🤝 Responds formally to user interactions (thank yous, FAQs)

---

## 🛠️ Running the Chatbot
1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/crime-prediction-api.git
   cd crime-prediction-api
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup environment variables:**
   - Copy `chatbot/.env.example` → rename to `.env`
   - Fill in required API keys.

4. **Launch the chatbot:**
   ```bash
   cd chatbot
   streamlit run chat_app.py
   ```

---

## 🚀 Future Enhancements
- Real-time crime and weather integration
- City-wide crime mapping (Folium/Leaflet)
- Predictive alerts based on user travel paths

---



