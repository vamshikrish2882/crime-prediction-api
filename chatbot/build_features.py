import pandas as pd

# Load raw dataset once at top level
raw_data = pd.read_csv("data/crime_climate_raw.csv")

def build_features(neighborhood, hour):
    """
    Build input features based on neighborhood and hour.
    """

    try:
        # Filter dataset
        match = raw_data[(raw_data["Neighborhood"].str.lower() == neighborhood.lower()) & (raw_data["Hour"] == hour)]

        if match.empty:
            return None  # No match found

        match = match.iloc[0]

        features = {
            "Hour": match["Hour"],
            "DayOfWeek": match["DayOfWeek"],
            "PremiseType": match["PremiseType"],
            "Weapon": match["Weapon"],
            "Race": match["Race"],
            "Gender": match["Gender"],
            "Sex": match["Gender"],  # Assuming Sex = Gender for simplicity
            "Age": match["Age"],
            "Latitude": match["Latitude"],
            "Longitude": match["Longitude"],
            "Year": match["Year"],
            "Month": match["Month"],
            "Total_Population": match["Total_Population"],
            "Male_Population": match["Male_Population"],
            "Female_Population": match["Female_Population"],
            "White_Population": match["White_Population"],
            "Black_Population": match["Black_Population"],
            "Asian_Population": match["Asian_Population"],
            "TwoOrMoreRaces_Population": match["TwoOrMoreRaces_Population"],
            "High_School_Grad": match["High_School_Grad"],
            "College_Grad": match["College_Grad"],
            "Median_Income": match["Median_Income"],
            "Labor_Force": match["Labor_Force"],
            "Unemployed": match["Unemployed"],
            "Temperature_C": match["Temperature_C"],
            "Dew_Point_C": match["Dew_Point_C"],
            "Wet_Bulb_Temperature_C": match["Wet_Bulb_Temperature_C"],
            "Relative_Humidity_2m": match["Relative_Humidity_2m"],
            "Specific_Humidity_2m": match["Specific_Humidity_2m"],
            "Precipitation_mm": match["Precipitation_mm"],
            "Pressure_hPa": match["Pressure_hPa"],
            "Wind_Speed_10m_mps": match["Wind_Speed_10m_mps"],
            "Wind_Speed_50m_mps": match["Wind_Speed_50m_mps"],
            "Solar_Radiation_Wm2": match["Solar_Radiation_Wm2"],
            "Weather_Category": match["Weather_Category"],
            "Income_Level": match["Income_Level"],
            "Employment_Status": match["Employment_Status"],
            "Majority_Race": match["Majority_Race"],
        }

        return features

    except Exception as e:
        print(f"❌ Error building features: {e}")
        return None
