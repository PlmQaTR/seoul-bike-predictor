import joblib
import streamlit as st
import numpy as np
import pandas as pd

## ---------------------------------------------------------
## Page config (must be first Streamlit call)
## ---------------------------------------------------------
st.set_page_config(
    page_title="Seoul Bike Demand Predictor",
    page_icon="🚲",
    layout="centered",
)

## ---------------------------------------------------------
## Load trained model
## ---------------------------------------------------------
model = joblib.load("seoul_bike_model.pkl")

## ---------------------------------------------------------
## Reference stats (from training data) used to contextualise
## the prediction for the user, e.g. "this is above average"
## ---------------------------------------------------------
TYPICAL_MEDIAN = 542
TYPICAL_HIGH = 1084   # 75th percentile
TYPICAL_MAX = 3556

## ---------------------------------------------------------
## Calm wallpaper: a very soft, low-opacity color wash layered
## ON TOP of Streamlit's own theme background (var(--background-color))
## rather than replacing it. This keeps the app's built-in
## light/dark text colors intact, so nothing goes invisible
## the way a full hardcoded background did before.
## ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(180deg, rgba(163, 201, 219, 0.16) 0%, rgba(163, 201, 219, 0.04) 55%, transparent 100%),
            var(--background-color);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

## ---------------------------------------------------------
## Banner image
## ---------------------------------------------------------
st.image(
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7h_26oHvEezqnfb--srXasGDLXg3BcRu1f7iZbL0zEw&s=10",
    use_container_width=True,
)

## ---------------------------------------------------------
## Header
## ---------------------------------------------------------
st.title("🚲 Seoul Bike Rental Demand Predictor")
st.caption(
    "Forecast hourly bike rental demand from weather and calendar conditions — "
    "built to help city operators plan bike availability ahead of time."
)

st.divider()

## ---------------------------------------------------------
## Helper: pairs a slider with a manual number-entry box so
## users can either drag or type an exact value. Both widgets
## share state via on_change callbacks, so editing either one
## keeps the other in sync.
## ---------------------------------------------------------
def slider_with_input(label, key, min_value, max_value, default, step, help=None):
    if key not in st.session_state:
        st.session_state[key] = default
    input_key = f"{key}_num"
    if input_key not in st.session_state:
        st.session_state[input_key] = default

    def _sync_from_slider():
        st.session_state[input_key] = st.session_state[key]

    def _sync_from_input():
        st.session_state[key] = st.session_state[input_key]

    slider_col, input_col = st.columns([3, 1])
    with slider_col:
        st.slider(
            label, min_value, max_value, step=step,
            key=key, on_change=_sync_from_slider, help=help,
        )
    with input_col:
        st.number_input(
            label, min_value=min_value, max_value=max_value, step=step,
            key=input_key, on_change=_sync_from_input,
            label_visibility="collapsed",
        )
    return st.session_state[key]


## ---------------------------------------------------------
## User inputs — centered in the main flow, no sidebar
## ---------------------------------------------------------
st.subheader("📅 Date & Time")
col1, col2 = st.columns(2)
with col1:
    date_selected = st.date_input("Date", help="The day you want to forecast demand for.")
with col2:
    hour_selected = slider_with_input(
        "Hour of day", "hour_slider", 0, 23, 8, 1,
        help="0 = midnight, 12 = noon, 23 = 11pm.",
    )

season_icons = {"Spring": "🌸", "Summer": "☀️", "Autumn": "🍂", "Winter": "❄️"}
season_selected = st.selectbox(
    "Season", list(season_icons.keys()),
    format_func=lambda s: f"{season_icons[s]} {s}",
)

st.subheader("🌤️ Weather Conditions")
w_col1, w_col2 = st.columns(2)
with w_col1:
    temperature_selected = slider_with_input(
        "Temperature (°C)", "temperature_slider", -18.0, 40.0, 13.5, 0.5,
    )
    dew_point_selected = slider_with_input(
        "Dew Point Temperature (°C)", "dew_point_slider", -31.0, 27.0, 4.7, 0.5,
    )
    is_raining_selected = st.toggle("🌧️ Raining right now?", value=False)
with w_col2:
    humidity_selected = slider_with_input(
        "Humidity (%)", "humidity_slider", 0, 100, 57, 1,
    )
    solar_radiation_selected = slider_with_input(
        "Solar Radiation (MJ/m²)", "solar_radiation_slider", 0.0, 3.5, 0.01, 0.05,
    )

st.write("")
predict_clicked = st.button("🔮 Predict Demand", use_container_width=True, type="primary")

## ---------------------------------------------------------
## Result
## ---------------------------------------------------------
if predict_clicked:
    with st.spinner("Running the model..."):
        ## Feature engineering (mirrors the training notebook
        ## exactly — must match model.feature_names_in_)
        month = date_selected.month
        day = date_selected.day
        is_weekend = date_selected.weekday() >= 5  # Sat/Sun

        ## Cyclical encodings (mirrors the training notebook exactly):
        ## Hour -> hour_sin/hour_cos (period 24), so 23:00 and 00:00 end up
        ## close together instead of far apart like raw hour values would.
        hour_sin = np.sin(2 * np.pi * hour_selected / 24)
        hour_cos = np.cos(2 * np.pi * hour_selected / 24)

        ## month -> month_sin/month_cos (period 12) and day -> day_sin/day_cos.
        ## Note: the training notebook divides day by 12 (not 31) when
        ## computing day_sin/day_cos, so that's reproduced here as-is to
        ## match what the model actually learned (model.feature_names_in_).
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_sin = np.sin(2 * np.pi * day / 12)
        day_cos = np.cos(2 * np.pi * day / 12)

        ## Rainfall -> is_raining ("Yes"/"No"), mirroring the training
        ## notebook: raw Rainfall(mm) is dropped in favor of a simple
        ## rain / no-rain flag, since demand craters whenever any rain
        ## is present regardless of amount.
        is_raining = "Yes" if is_raining_selected else "No"

        df_input = pd.DataFrame({
            "Temperature(°C)": [temperature_selected],
            "Humidity(%)": [humidity_selected],
            "Dew point temperature(°C)": [dew_point_selected],
            "Solar Radiation (MJ/m2)": [solar_radiation_selected],
            "is_weekend": [is_weekend],
            "hour_sin": [hour_sin],
            "hour_cos": [hour_cos],
            "month_sin": [month_sin],
            "month_cos": [month_cos],
            "day_sin": [day_sin],
            "day_cos": [day_cos],
            "Seasons": [season_selected],
            "is_raining": [is_raining],
        })

        df_input = pd.get_dummies(df_input)
        df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

        y_pred = model.predict(df_input)[0]
        y_pred = int(max(0, round(y_pred)))

    ## Classify demand level relative to the training data's distribution
    if y_pred < TYPICAL_MEDIAN:
        level, emoji = "Low demand", "🟢"
    elif y_pred < TYPICAL_HIGH:
        level, emoji = "Moderate demand", "🟡"
    else:
        level, emoji = "High demand", "🔴"

    st.divider()

    result_col, context_col = st.columns([2, 3])
    with result_col:
        st.metric(
            label=f"{emoji} Predicted Bikes — {date_selected.strftime('%d %b %Y')}, {hour_selected:02d}:00",
            value=f"{y_pred:,} bikes",
            delta=f"{y_pred - TYPICAL_MEDIAN:+,} vs. typical hour",
        )
        st.caption(f"**{level}** for this hour, based on similar historical conditions.")

    with context_col:
        st.write("**How this compares to a typical hour:**")
        progress_val = min(y_pred / TYPICAL_MAX, 1.0)
        st.progress(progress_val)
        st.caption(f"0 (quietest hour) — {TYPICAL_MAX:,} bikes (busiest hour on record)")

    with st.expander("📊 See the exact inputs sent to the model"):
        display_df = df_input.copy()
        display_df.index = ["Value"]
        st.dataframe(display_df.T, use_container_width=True)

    st.caption(
        "💡 Tip: try switching only the Hour slider between 8am and 6pm to see how "
        "commute-time demand differs — useful for planning bike redistribution."
    )