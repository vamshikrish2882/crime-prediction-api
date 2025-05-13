# 🧠 Building an AI-Powered Crime Prediction Chatbot: A Real-Time Public Safety Companion

## 🧭 Introduction

As urban environments grow more complex, safety and situational awareness have become top priorities for residents, travelers, and policymakers alike. To address this, I developed an AI-powered chatbot that predicts the **most likely type of crime** for a given location and time. This chatbot serves not only as an interactive public safety tool, but also as a real-world application of machine learning, natural language processing, and civic data integration.

The chatbot takes natural-language queries like “Is it safe in Downtown at 11PM?”, extracts the necessary information (time and place), generates contextual features from weather, socio-economic, and location datasets, and delivers an insightful prediction along with safety tips and historical trends. This project exemplifies how data science can move beyond dashboards and into real-time, public-facing applications.

---

## 🎯 Project Objective

The primary objective was to create a tool that simplifies crime risk assessment for the average user. Unlike static dashboards, this chatbot uses conversational AI to:
- Interpret user intent through natural-language inputs
- Detect time and location entities with an LLM-based extractor
- Generate features using real-time weather and demographic information
- Predict the most probable crime category with an XGBoost model
- Display contextual insights and safety guidance

This project demonstrates practical applications of machine learning in the civic tech space, with an emphasis on responsible, explainable AI.

---

## 🧾 Data Collection and Feature Engineering

To train the model and generate useful predictions, I combined three major datasets:

- 🕵️‍♂️ **Baltimore Police Crime Records (2011–2024)**: Includes crime category, location, time of occurrence, and weapon involved.
- 🌦️ **NASA POWER Weather Dataset**: Hourly weather conditions such as temperature, dew point, humidity, wind speed, and precipitation.
- 🧑‍🏫 **Census ACS 5-Year Demographic Data**: Unemployment rates, income levels, education rates, and population breakdown by race/ethnicity for each neighborhood.

Using the `build_features.py` module, I engineered a robust feature set of over 37 variables:
- **Temporal**: Hour of day, day of week, month, season
- **Weather**: Temperature, humidity, wind speed, dew point
- **Location**: Latitude, longitude, neighborhood
- **Demographics**: Unemployment, median income, race percentages
- **Crime Context**: Premise type, historical hotspot classification

These features are merged into a final dataframe used for model training and live prediction.

---

## 🤖 Model Training and Performance (XGBoost)

I used **XGBoost**, a high-performance gradient-boosted tree algorithm, to classify crime types into five categories (e.g., Assault, Theft, Robbery).

### Training Workflow:
- Cleaned and merged all datasets into a master feature matrix
- Performed one-hot encoding and scaling on categorical and numerical features
- Used an 80/20 stratified split to maintain class distribution
- Tuned hyperparameters with `GridSearchCV`

### Model Metrics:
- **Accuracy**: `75.99%` (Validation)
- **Macro F1-score**: Balanced performance across all categories

![Model Performance](assets/model_performance.png)

This model outperformed simpler baselines like decision trees or logistic regression, especially in handling class imbalance and high-dimensional features.

---

### 🔧 Best Model Parameters

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
)
```

These parameters were selected to strike a balance between model accuracy and generalization.

---

## 🧠 Natural Language Understanding (LLM-Powered)

To support real-time conversation, I built an NLP pipeline using an **LLM from OpenRouter API**.

- Parses user input (e.g., "Should I go to Charles Village tonight?")
- Extracts structured `location` and `time` entities
- Converts vague terms like “midnight” or “tomorrow” into datetime objects

If no valid location/time is detected, the system defaults to safe fallback values or requests clarification.

---

## 💬 Streamlit-Based Chat Interface

The chatbot runs on **Streamlit**, providing a responsive interface that allows users to ask questions and receive predictions in real time. It offers:
- Natural language input field
- Predicted crime category (e.g., “Assault”)
- Safety tip tailored to severity level
- Pie chart visualization of historical crime distribution in that area

![Chatbot Interface](assets/Chatbot_inteface.png)
![Pie Chart](assets/piechart.png)

---

## 🧱 Backend Architecture

The application uses a modular backend for scalability and ease of updates.

![Architecture Diagram](assets/flow_chart.png)

- `chat_app.py` (Streamlit Frontend)
- `extract_location_time.py` (LLM Interface)
- `build_features.py` (Data Transformation)
- `XGBoost-model.ipynb` (Model Training)
- Flask REST API (Model Prediction Endpoint)

The separation of concerns allows for independent upgrades to each layer.

---

## 🚧 Challenges and Solutions

### Ambiguous Inputs
Handled using structured prompts and fallback defaults when user intent isn’t clear.

### Model Bias
Balanced training set using stratification and monitored fairness across demographic features.

### Deployment Issues
Secured environment variables with `.env`, handled API timeouts and validated prediction payloads.

---

## 🔮 Future Work

- Add **Folium map visualizations** for crime hotspots
- Integrate live crime alerts using **open police APIs**
- Enable **voice input** or **SMS access** for wider accessibility
- Implement **feedback learning loop** to refine model

---

## 🧩 Conclusion

This project is a real-world demonstration of how AI can support public safety without requiring users to interpret raw data. By using weather, time, location, and social context, the chatbot offers tailored predictions and practical tips. It also shows how academic knowledge—like model evaluation, feature engineering, and NLP—can be deployed responsibly to solve real-world problems.

The **Crime Prediction Chatbot** is not just a prototype—it is a scalable framework that could be expanded to serve multiple cities and stakeholders in the near future.
