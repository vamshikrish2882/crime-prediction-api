# 🧠 Building an AI-Powered Crime Prediction Chatbot: A Real-Time Public Safety Companion

## 🧭 Introduction

In an era where data and AI increasingly shape how cities operate, ensuring public safety through intelligent systems has become not just possible—but essential. This project presents the design and implementation of an AI-driven chatbot that predicts the **type of crime likely to occur based on location and time**, while also providing practical safety recommendations. Focused on Baltimore, the chatbot draws from historical crime data, weather conditions, and demographic indicators to deliver actionable insights to everyday users.

The system combines **machine learning**, **natural language processing**, and **real-time deployment technologies** to answer one question:  
**_"Is it safe to go to this area at this time?"_**  
With a user-friendly interface, this chatbot brings data-driven crime awareness directly to the public.

---

## 🎯 Project Objective

Traditional crime dashboards often present static, historical data that can be hard to interpret. This chatbot streamlines user interaction by:
- Accepting **natural-language queries** like *“Is it safe in Charles Village at 2AM?”*
- Extracting structured **location and time** from the input
- Generating a **context-rich feature vector** using demographic and weather data
- Predicting the **most likely crime type**
- Delivering **safety tips** and **historical crime stats**

---

## 🧾 Data Collection and Feature Engineering

Three primary datasets were used:
- 🕵️‍♂️ **Baltimore Police crime records(2011-2024)**
- 🌦️ **NASA POWER weather reports**
- 🧑‍🏫 **Neighborhood demographic data - ACS 5 year**

The `build_features.py` module transforms these into 37+ features:
- **Time-based**: Hour, Day of Week, Month, Year  
- **Weather**: Temperature, Humidity, Wind Speed, Dew Point  
- **Demographics**: Unemployment, Education, Income, Race %  
- **Geographic**: Neighborhood, Latitude, Longitude  
- **Crime Context**: Premise Type, Weapon Involved

This dynamic pipeline ensures predictions are grounded in real-world, real-time factors.

---

## 🤖 Model Training: XGBoost Classifier

### Training Process:
- Cleaned and merged datasets
- Encoded categorical features
- Applied `GridSearchCV` for hyperparameter tuning
- Used an 80/20 stratified split to preserve label distribution

### Performance:
- **Validation Accuracy**: `75.99%`
- **Macro F1-scores** showed balanced prediction across multiple crime types

XGBoost was chosen for its robustness on structured data and support for class imbalance.

---

### 🔧 Best Model Parameters

The best configuration found during tuning was:

```python
XGBClassifier(
    subsample=0.8,
    n_estimators=300,
    min_child_weight=1,
    max_depth=10,
    learning_rate=0.1,
    gamma=1,
    colsample_bytree=0.6,
    objective='multi:softprob',
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)```
These parameters allowed the model to efficiently capture interactions among weather, time, and socioeconomic factors while minimizing overfitting.

---

### 🧠 Natural Language Input Handling

Users communicate naturally with the chatbot (e.g., *“What about Inner Harbor at midnight?”*).  
The `extract_location_time.py` script uses a **Large Language Model (LLM)** via **OpenRouter API** to extract:

- 📍 **Location**
- 🕑 **Time** (converted to 24-hour format)

Even vague inputs are parsed reliably using structured prompt engineering. If no valid location/time is detected, the system gracefully falls back on defaults.

---

### 💬 Streamlit-Based Chat Interface

The chatbot frontend is built with **Streamlit** in `chat_app.py`. Users receive:

- A greeting and usage instructions  
- Natural-language input field  
- **Predicted crime type** for that time/place  
- A **safety tip**  
- A **pie chart** showing historical top 5 crime types  

The system maintains chat history with `st.session_state`, creating a fluid conversation.

---

### 🧱 Backend Architecture

The app runs through a **Flask API** hosted on **Render.com**, keeping the system modular and scalable.

- **Frontend**: Streamlit interface  
- **Model Server**: Flask REST API for prediction  
- **LLM NLP**: OpenRouter for input parsing  
- **Feature Builder**: `build_features.py`  
- **Visualization**: `matplotlib` for charts  

This architecture supports decoupled scaling, fast prototyping, and easy updates.

---

### 🚧 Challenges and Solutions

🔸 **Ambiguous Inputs**  
Handled using LLM parsing + fallbacks.

🔸 **Model Bias**  
Stratified sampling + transparent display of prediction logic.

🔸 **Deployment Errors**  
API keys, environment variables, and response handling were managed using `.env` and `requests`.

---

### 🔮 Future Enhancements

- 🗺️ **Interactive heatmaps** (Folium)  
- 📡 **Live crime API integration**  
- 🧠 **Feedback loop for model improvement**  
- 🗣️ **Voice assistant or SMS integration**  
- 🌍 **Expansion to other cities**

---

### 🧩 Conclusion

This chatbot blends **machine learning**, **weather analysis**, **demographics**, and **natural language understanding** to empower users with real-time crime awareness. It demonstrates how AI can be designed not just to automate, but to protect, inform, and serve communities.

From curious citizens to city planners, the **Crime Prediction Chatbot** is a modern tool for **data-informed urban safety**—one question at a time.
