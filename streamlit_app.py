import joblib
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

## Page config (wide layout so two scenarios can sit side by side)
st.set_page_config(page_title="Seoul Bike Rental Predictor", page_icon="🚲", layout="wide")

## A little bit of custom styling to make the app feel more polished
st.markdown("""
    <style>
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

## Load trained model
model = joblib.load("seoul_bike_model.pkl")

## Streamlit app
st.title("🚲 Seoul Bike Rental Demand Prediction")
st.write("Predict the number of rented bikes based on date, time and weather conditions. "
         "Adjust the inputs below and click **Predict** to see the results.")

## Session state for history and comparison mode
if "history" not in st.session_state:
    st.session_state.history = []


def build_features(date_selected, hour_selected, season_selected, temp_selected,
                    dew_point_selected, humidity_selected, solar_radiation_selected,
                    is_raining_selected):
    """Build a single-row feature DataFrame matching the training pipeline."""
    is_weekend = date_selected.weekday() >= 5
    month = date_selected.month
    day = date_selected.day

    ## Cyclical encoding for hour, month and day
    hour_sin = np.sin(2 * np.pi * hour_selected / 24)
    hour_cos = np.cos(2 * np.pi * hour_selected / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_sin = np.sin(2 * np.pi * day / 12)
    day_cos = np.cos(2 * np.pi * day / 12)

    df_input = pd.DataFrame({
        'Temperature(°C)': [temp_selected],
        'Humidity(%)': [humidity_selected],
        'Dew point temperature(°C)': [dew_point_selected],
        'Solar Radiation (MJ/m2)': [solar_radiation_selected],
        'is_weekend': [is_weekend],
        'hour_sin': [hour_sin],
        'hour_cos': [hour_cos],
        'month_sin': [month_sin],
        'month_cos': [month_cos],
        'day_sin': [day_sin],
        'day_cos': [day_cos],
        'Seasons': [season_selected],
        'is_raining': [is_raining_selected],
    })

    
    df_input = pd.get_dummies(df_input)

    ## Align columns with what the model was trained on
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)
    return df_input


def predict(df_input):
    return model.predict(df_input)[0]


def synced_slider(label, min_value, max_value, value, key_prefix, step=None, fmt=None, help=None):
    slider_key = f"{key_prefix}_slider"
    input_key = f"{key_prefix}_input"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = value
    if input_key not in st.session_state:
        st.session_state[input_key] = value

    def sync_from_slider():
        st.session_state[input_key] = st.session_state[slider_key]

    def sync_from_input():
        st.session_state[slider_key] = st.session_state[input_key]

    slider_col, input_col = st.columns([4, 1.3])
    with slider_col:
        st.slider(label, min_value=min_value, max_value=max_value, step=step,
                  key=slider_key, on_change=sync_from_slider, help=help)
    with input_col:
        st.number_input(label, min_value=min_value, max_value=max_value, step=step,
                         key=input_key, on_change=sync_from_input,
                         label_visibility="collapsed", format=fmt)

    return st.session_state[slider_key]


def render_inputs(key_prefix, label=None):
    """Render the input widgets for one scenario and return the raw values."""
    if label:
        st.subheader(f"📋 {label}")

    st.markdown("**📅 Date & Time**")
    date_selected = st.date_input(
        "Select Date", value=datetime(2018, 6, 1), key=f"{key_prefix}_date",
        help="The date on which the predicted rental takes place."
    )
    hour_selected = synced_slider(
        "Select Hour of Day", min_value=0, max_value=23, value=8,
        key_prefix=f"{key_prefix}_hour", step=1,
        help="Hour of the day in 24-hour format. (23 -> 11pm, 0-> 12am)"
    )
    season_selected = st.selectbox(
        "Select Season", ["Spring", "Summer", "Autumn", "Winter"], key=f"{key_prefix}_season",
        help="The season for the selected date."
    )

    st.markdown("**🌤️ Weather Conditions**")
    temp_selected = synced_slider(
        "Temperature (°C)", min_value=-20.0, max_value=40.0, value=15.0,
        key_prefix=f"{key_prefix}_temp", step=0.1, fmt="%.1f",
        help="Seoul surrounding temperature in degrees Celsius at the time of rental."
    )
    dew_point_selected = synced_slider(
        "Dew Point Temperature (°C)", min_value=-30.0, max_value=30.0, value=10.0,
        key_prefix=f"{key_prefix}_dew", step=0.1, fmt="%.1f",
        help="Temperature at which air becomes saturated with moisture."
    )
    humidity_selected = synced_slider(
        "Humidity (%)", min_value=0, max_value=100, value=50,
        key_prefix=f"{key_prefix}_humidity", step=1,
        help="Humidity of Seoul as a percentage (0-100)."
    )
    solar_radiation_selected = synced_slider(
        "Solar Radiation (MJ/m2)", min_value=0.0, max_value=4.0, value=0.5,
        key_prefix=f"{key_prefix}_solar", step=0.01, fmt="%.2f",
        help="Solar radiation intensity in megajoules per square meter."
    )
    is_raining_selected = st.selectbox(
        "Is it raining? 🌧️", ["No", "Yes"], key=f"{key_prefix}_rain",
        help="Whether is there any rainfall at the time of rental."
    )

    return dict(
        date_selected=date_selected,
        hour_selected=hour_selected,
        season_selected=season_selected,
        temp_selected=temp_selected,
        dew_point_selected=dew_point_selected,
        humidity_selected=humidity_selected,
        solar_radiation_selected=solar_radiation_selected,
        is_raining_selected=is_raining_selected,
    )


def add_to_history(label, inputs, y_pred):
    st.session_state.history.append({
        "Scenario": label,
        "Date": inputs["date_selected"].isoformat(),
        "Hour": inputs["hour_selected"],
        "Season": inputs["season_selected"],
        "Temp (°C)": inputs["temp_selected"],
        "Humidity (%)": inputs["humidity_selected"],
        "Raining": inputs["is_raining_selected"],
        "Predicted Count": round(y_pred),
        "Predicted At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


## Mode selector
mode = st.radio(
    "Choose a mode",
    ["🔮 Single Prediction", "⚖️ Compare Two Scenarios"],
    horizontal=True,
    help="Predict one scenario, or compare two side by side."
)
mode = "Single Prediction" if "Single" in mode else "Compare Two Scenarios"

st.divider()

if mode == "Single Prediction":
    inputs = render_inputs("single")

    if st.button("🔮 Predict Bike Rental Count", type="primary"):
        df_input = build_features(**inputs)
        y_pred = predict(df_input)

        result_col, _ = st.columns([1, 2])
        with result_col:
            st.metric("Predicted Rented Bike Count", f"{y_pred:,.0f} bikes")
        st.success("Prediction complete! Details saved to history below. ✅")

        add_to_history("Single", inputs, y_pred)

else:
    col_a, col_b = st.columns(2)

    with col_a:
        inputs_a = render_inputs("scenario_a", label="Scenario A")

    with col_b:
        inputs_b = render_inputs("scenario_b", label="Scenario B")

    if st.button("⚖️ Compare Predictions", type="primary"):
        df_a = build_features(**inputs_a)
        df_b = build_features(**inputs_b)
        y_pred_a = predict(df_a)
        y_pred_b = predict(df_b)

        diff = y_pred_a - y_pred_b

        result_a, result_b = st.columns(2)
        with result_a:
            st.metric("Scenario A", f"{y_pred_a:,.0f} bikes",
                      delta=f"{diff:+,.0f} vs Scenario B")
        with result_b:
            st.metric("Scenario B", f"{y_pred_b:,.0f} bikes",
                      delta=f"{-diff:+,.0f} vs Scenario A")
        st.success("Comparison complete! Details saved to history below. ✅")

        add_to_history("Scenario A", inputs_a, y_pred_a)
        add_to_history("Scenario B", inputs_b, y_pred_b)

## Prediction history
st.divider()
st.subheader("🕒 Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No predictions yet. Run a prediction above to see it logged here.")