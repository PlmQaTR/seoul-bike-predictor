import joblib
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

wide_layout = st.session_state.get("compare_mode", False)
st.set_page_config(page_title="Seoul Bike Demand Predictor", page_icon="🚲", layout="wide" if wide_layout else "centered")

model = joblib.load("seoul_bike_model.pkl")

TYPICAL_MEDIAN = 542
TYPICAL_HIGH = 1084
TYPICAL_MAX = 3556
MODEL_RMSE = 150

LEVEL_COLORS = {
    "Low demand": ("seagreen", "color-mix(in srgb, seagreen 12%, transparent)"),
    "Moderate demand": ("mediumseagreen", "color-mix(in srgb, mediumseagreen 12%, transparent)"),
    "High demand": ("darkgreen", "color-mix(in srgb, darkgreen 12%, transparent)"),
}

# result card sizing for the two-column compare layout vs the full-width single view
COMPACT_SIZES = {"header": 15, "value": 46, "unit": 20, "pad": "18px 20px", "pill": 14, "pill_pad": "4px 12px", "delta": 15, "svg": 260}
FULL_SIZES = {"header": 18, "value": 68, "unit": 32, "pad": "30px 34px", "pill": 16, "pill_pad": "5px 16px", "delta": 17, "svg": 340}

st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(180deg, rgba(163, 201, 219, 0.16) 0%, rgba(163, 201, 219, 0.04) 55%, transparent 100%),
            var(--background-color);
    }
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }
    .banner-img img {
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="banner-img">', unsafe_allow_html=True)
st.image(
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7h_26oHvEezqnfb--srXasGDLXg3BcRu1f7iZbL0zEw&s=10",
    use_container_width=True,
)
st.markdown('</div>', unsafe_allow_html=True)

st.title("🚲 Seoul Bike Rental Demand Predictor")
st.caption("Forecast hourly bike rental demand from weather and calendar conditions, so city operators can plan bike availability ahead of time.")
st.divider()


def slider_with_input(label, key, min_value, max_value, default, step, help=None, stacked=False):
    if key not in st.session_state:
        st.session_state[key] = default
    input_key = key + "_num"
    if input_key not in st.session_state:
        st.session_state[input_key] = default

    # keep slider and number box in sync without fighting each other on rerun
    sync_from_slider = lambda: st.session_state.update({input_key: st.session_state[key]})
    sync_from_input = lambda: st.session_state.update({key: st.session_state[input_key]})

    if stacked:
        st.slider(label, min_value, max_value, step=step, key=key, on_change=sync_from_slider, help=help)
        num_col, _ = st.columns([1, 2])
        with num_col:
            st.number_input(label, min_value, max_value, step=step, key=input_key,
                             on_change=sync_from_input, label_visibility="collapsed")
    else:
        slider_col, input_col = st.columns([3, 1])
        with slider_col:
            st.slider(label, min_value, max_value, step=step, key=key, on_change=sync_from_slider, help=help)
        with input_col:
            st.number_input(label, min_value, max_value, step=step, key=input_key,
                             on_change=sync_from_input, label_visibility="collapsed")

    return st.session_state[key]


def render_input_panel(prefix="", compact=False):
    heading = (lambda t: st.markdown(f"**{t}**")) if compact else st.subheader

    heading("📅 Date & Time")
    col1, col2 = st.columns(2)
    with col1:
        date_selected = st.date_input("Date", key=prefix + "date", help="The day you want to forecast demand for.")
    with col2:
        hour_selected = slider_with_input(
            "Hour of day", prefix + "hour_slider", 0, 23, 8, 1,
            help="0 = midnight, 12 = noon, 23 = 11pm.", stacked=compact,
        )

    season_icons = {"Spring": "🌸", "Summer": "☀️", "Autumn": "🍂", "Winter": "❄️"}
    season_selected = st.selectbox(
        "Season", list(season_icons.keys()),
        format_func=lambda s: f"{season_icons[s]} {s}", key=prefix + "season",
    )

    heading("🌤️ Weather Conditions")
    w_col1, w_col2 = st.columns(2)
    with w_col1:
        temperature_selected = slider_with_input("Temperature (°C)", prefix + "temp_slider", -18.0, 40.0, 13.5, 0.5, stacked=compact)
        dew_point_selected = slider_with_input("Dew Point Temperature (°C)", prefix + "dew_slider", -31.0, 27.0, 4.7, 0.5, stacked=compact)
        is_raining_selected = st.toggle("🌧️ Raining right now?", value=False, key=prefix + "is_raining")
    with w_col2:
        humidity_selected = slider_with_input("Humidity (%)", prefix + "humidity_slider", 0, 100, 57, 1, stacked=compact)
        solar_radiation_selected = slider_with_input("Solar Radiation (MJ/m²)", prefix + "solar_slider", 0.0, 3.5, 0.01, 0.05, stacked=compact)

    return {
        "date_selected": date_selected,
        "hour_selected": hour_selected,
        "season_selected": season_selected,
        "temperature_selected": temperature_selected,
        "dew_point_selected": dew_point_selected,
        "is_raining_selected": is_raining_selected,
        "humidity_selected": humidity_selected,
        "solar_radiation_selected": solar_radiation_selected,
    }


compare_mode = st.toggle(
    "🔀 Compare two scenarios (A vs B)", key="compare_mode",
    help="Set up two independent scenarios and see their predicted demand side-by-side.",
)

if compare_mode:
    st.caption("Adjust Scenario A and Scenario B independently, then compare their predicted demand at a glance.")
    colA, colB = st.columns(2, gap="large")
    with colA:
        with st.container(border=True):
            st.subheader("🅰️ Scenario A")
            inputs_A = render_input_panel(prefix="A_", compact=True)
    with colB:
        with st.container(border=True):
            st.subheader("🅱️ Scenario B")
            inputs_B = render_input_panel(prefix="B_", compact=True)
    st.write("")
    predict_clicked = st.button("🔮 Predict Both Scenarios", use_container_width=True, type="primary")
else:
    with st.container(border=True):
        inputs = render_input_panel(prefix="")
    st.write("")
    predict_clicked = st.button("🔮 Predict Demand", use_container_width=True, type="primary")


def compute_prediction(inputs):
    date_selected = inputs["date_selected"]
    hour_selected = inputs["hour_selected"]
    month, day = date_selected.month, date_selected.day
    is_weekend = date_selected.weekday() >= 5

    hour_sin, hour_cos = np.sin(2 * np.pi * hour_selected / 24), np.cos(2 * np.pi * hour_selected / 24)
    month_sin, month_cos = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    day_sin, day_cos = np.sin(2 * np.pi * day / 12), np.cos(2 * np.pi * day / 12)
    is_raining = "Yes" if inputs["is_raining_selected"] else "No"

    df_input = pd.DataFrame({
        "Temperature(°C)": [inputs["temperature_selected"]],
        "Humidity(%)": [inputs["humidity_selected"]],
        "Dew point temperature(°C)": [inputs["dew_point_selected"]],
        "Solar Radiation (MJ/m2)": [inputs["solar_radiation_selected"]],
        "is_weekend": [is_weekend],
        "hour_sin": [hour_sin], "hour_cos": [hour_cos],
        "month_sin": [month_sin], "month_cos": [month_cos],
        "day_sin": [day_sin], "day_cos": [day_cos],
        "Seasons": [inputs["season_selected"]],
        "is_raining": [is_raining],
    })
    df_input = pd.get_dummies(df_input).reindex(columns=model.feature_names_in_, fill_value=0)
    y_pred = int(max(0, round(model.predict(df_input)[0])))

    if y_pred < TYPICAL_MEDIAN:
        level, emoji = "Low demand", "🟢"
    elif y_pred < TYPICAL_HIGH:
        level, emoji = "Moderate demand", "🟡"
    else:
        level, emoji = "High demand", "🔴"

    return {
        "y_pred": y_pred, "level": level, "emoji": emoji,
        "delta": y_pred - TYPICAL_MEDIAN,
        "date_label": date_selected.strftime('%d %b %Y'),
        "hour_label": f"{hour_selected:02d}:00",
        "rotation": min(y_pred / TYPICAL_MAX, 1.0) * 360,
    }


def log_history(inputs, result, scenario="-"):
    st.session_state.setdefault("history", [])
    st.session_state["history"].insert(0, {
        "Predicted at": datetime.now().strftime("%H:%M:%S"),
        "Scenario": scenario,
        "Forecast date": result["date_label"],
        "Hour": result["hour_label"],
        "Temp (°C)": inputs["temperature_selected"],
        "Humidity (%)": inputs["humidity_selected"],
        "Predicted demand": result["y_pred"],
        "Level": f"{result['emoji']} {result['level']}",
    })
    st.session_state["history"] = st.session_state["history"][:25]


if predict_clicked:
    with st.spinner("Running the model..."):
        if compare_mode:
            result_A, result_B = compute_prediction(inputs_A), compute_prediction(inputs_B)
            st.session_state["result_A"], st.session_state["result_B"] = result_A, result_B
            log_history(inputs_A, result_A, scenario="A")
            log_history(inputs_B, result_B, scenario="B")
        else:
            result = compute_prediction(inputs)
            st.session_state["result"] = result
            log_history(inputs, result, scenario="-")


def render_result(result, is_fresh, key_suffix="", heading=None, compact=False):
    y_pred, level, emoji, delta = result["y_pred"], result["level"], result["emoji"], result["delta"]
    delta_sign = "+" if delta >= 0 else ""

    seq_key = f"_anim_seq{key_suffix}"
    st.session_state[seq_key] = st.session_state.get(seq_key, 0) + 1
    anim_tag = f"{key_suffix}{st.session_state[seq_key]}"

    if heading:
        st.markdown(f"##### {heading}")

    accent, accent_bg = LEVEL_COLORS[level]
    sizes = COMPACT_SIZES if compact else FULL_SIZES
    header_fs, value_fs, unit_fs = sizes["header"], sizes["value"], sizes["unit"]
    pad, pill_fs, pill_pad, delta_fs, svg_width = sizes["pad"], sizes["pill"], sizes["pill_pad"], sizes["delta"], sizes["svg"]

    badge_anim_name = f"badgeIn{anim_tag}"
    badge_anim = f"animation: {badge_anim_name} 0.35s ease-out both;" if is_fresh else ""

    st.markdown(f"""
        <style>
        @keyframes {badge_anim_name} {{
            from {{ opacity: 0; transform: scale(0.97) translateY(4px); }}
            to {{ opacity: 1; transform: scale(1) translateY(0); }}
        }}
        </style>
        <div style="background: {accent_bg}; border: 1px solid {accent}; border-radius: 18px; padding: {pad}; margin-bottom: 20px; box-shadow: 0 6px 20px rgba(0,0,0,0.10);{badge_anim}">
            <div style="font-size: {header_fs}px; opacity: 0.75; margin-bottom: 8px;">{emoji} Predicted demand for {result['date_label']}, {result['hour_label']}</div>
            <div style="font-size: {value_fs}px; font-weight: 800; line-height: 1.1; letter-spacing: -0.5px;">{y_pred:,} <span style="font-size: {unit_fs}px; font-weight: 500; opacity: 0.7;">bikes</span></div>
            <div style="margin-top: 12px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                <span style="background: {accent}; color: white; padding: {pill_pad}; border-radius: 999px; font-weight: 700; font-size: {pill_fs}px;">{level}</span>
                <span style="opacity: 0.8; font-size: {delta_fs}px;">{delta_sign}{delta:,} vs. typical hour ({TYPICAL_MEDIAN:,})</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if not compact:
        st.markdown("<div style='font-size:17px; font-weight:700; margin-bottom:4px;'>How this compares to a typical hour:</div>", unsafe_allow_html=True)

    calm_w = min(TYPICAL_MEDIAN / TYPICAL_MAX, 1.0) * 100
    typical_w = min(TYPICAL_HIGH / TYPICAL_MAX, 1.0) * 100 - calm_w
    busy_w = 100 - calm_w - typical_w

    marker_to = min(result["y_pred"] / TYPICAL_MAX, 1.0) * 100
    marker_key = f"marker_displayed_pct{key_suffix}"
    marker_from = st.session_state.get(marker_key, 0.0)
    st.session_state[marker_key] = marker_to

    marker_anim_name = f"markerSlide{anim_tag}"
    marker_keyframes = f"@keyframes {marker_anim_name} {{ from {{ left: {marker_from:.1f}%; }} to {{ left: {marker_to:.1f}%; }} }}"
    marker_style = f"animation: {marker_anim_name} 0.9s cubic-bezier(0.34, 1, 0.4, 1) forwards;"

    if is_fresh:
        fill_anim_name = f"barGrow{anim_tag}"
        fill_anim_css = f"transform: scaleX(0); animation: {fill_anim_name} 0.9s ease-out forwards;"
        fill_keyframes = f"@keyframes {fill_anim_name} {{ to {{ transform: scaleX(1); }} }}"
    else:
        # no click this run, just redraw at rest so the meter doesn't replay the animation
        fill_anim_css = "transform: scaleX(1);"
        fill_keyframes = ""

    gauge_class = f"gauge-wrap{anim_tag}"
    bar_h = 22 if compact else 28
    st.markdown(f"""
        <style>
        .{gauge_class} {{ max-width: {svg_width + 60}px; margin: 10px auto 0 auto; }}
        .{gauge_class} .gauge-seg {{ transform-origin: left; {fill_anim_css} }}
        .{gauge_class} .gauge-seg.calm {{ animation-delay: 0s; }}
        .{gauge_class} .gauge-seg.typical {{ animation-delay: 0.15s; }}
        .{gauge_class} .gauge-seg.busy {{ animation-delay: 0.3s; }}{fill_keyframes}
        .{gauge_class} .gauge-marker {{ position: absolute; top: -14px; {marker_style} }}
        {marker_keyframes}
        </style>
        <div class="{gauge_class}">
        <div style="position: relative;">
            <div style="display:flex; height:{bar_h}px; border-radius:999px; overflow:hidden;">
                <div class="gauge-seg calm" style="flex:0 0 {calm_w:.2f}%; background:#2dd4bf;"></div>
                <div class="gauge-seg typical" style="flex:0 0 {typical_w:.2f}%; background:#f59e0b;"></div>
                <div class="gauge-seg busy" style="flex:0 0 {busy_w:.2f}%; background:#f87171;"></div>
            </div>
            <div class="gauge-marker" style="transform: translateX(-50%);">
                <div style="width:0; height:0; margin:0 auto; border-left:9px solid transparent; border-right:9px solid transparent; border-top:12px solid #a3e635;"></div>
                <div style="width:4px; height:{bar_h}px; background:#a3e635; margin:0 auto; border-radius:2px;"></div>
            </div>
        </div>
        </div>
        <div style="display:flex; justify-content:center; gap:{16 if compact else 26}px; margin-top:8px; font-size:{15 if compact else 16}px; font-weight:600; opacity:0.9;">
            <span>🟢 Calm</span><span>🟠 Typical</span><span>🔴 Busy</span>
        </div>
    """, unsafe_allow_html=True)

    if not compact:
        st.markdown(f"<div style='font-size:14px; opacity:0.75;'>0 (quietest hour) to {TYPICAL_MAX:,} bikes (busiest hour on record)</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:14px; opacity:0.75;'>📏 Model accuracy: predictions are typically within <b>±{MODEL_RMSE} bikes</b> (RMSE on held-out test data).</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px; opacity:0.75;'>💡 Tip: try switching only the Hour slider between 8am and 6pm to see how commute-time demand differs. Useful for planning bike redistribution.</div>", unsafe_allow_html=True)


if compare_mode:
    if "result_A" in st.session_state or "result_B" in st.session_state:
        st.divider()
        st.write("**Scenario comparison:**")
        gcolA, gcolB = st.columns(2, gap="large")
        with gcolA:
            if "result_A" in st.session_state:
                with st.container(border=True):
                    render_result(st.session_state["result_A"], is_fresh=predict_clicked, key_suffix="_A", heading="🅰️ Scenario A", compact=True)
        with gcolB:
            if "result_B" in st.session_state:
                with st.container(border=True):
                    render_result(st.session_state["result_B"], is_fresh=predict_clicked, key_suffix="_B", heading="🅱️ Scenario B", compact=True)

        if "result_A" in st.session_state and "result_B" in st.session_state:
            pred_A, pred_B = st.session_state["result_A"]["y_pred"], st.session_state["result_B"]["y_pred"]
            diff = pred_B - pred_A
            if diff == 0:
                st.info("🅰️ Scenario A and 🅱️ Scenario B predict the **same** demand.")
            else:
                higher, lower = ("B", "A") if diff > 0 else ("A", "B")
                base = pred_A if higher == "B" else pred_B
                pct = (abs(diff) / base * 100) if base else 0
                st.info(f"**Scenario {higher}** predicts **{abs(diff):,} more bikes** than **Scenario {lower}** (about {pct:.0f}% higher).")

        st.caption(f"0 (quietest hour) to {TYPICAL_MAX:,} bikes (busiest hour on record)")
        st.caption(f"📏 Model accuracy: predictions are typically within **±{MODEL_RMSE} bikes** (RMSE on held-out test data), for either scenario.")
else:
    if "result" in st.session_state:
        st.divider()
        with st.container(border=True):
            render_result(st.session_state["result"], is_fresh=predict_clicked, key_suffix="")


if st.session_state.get("history"):
    st.divider()
    hist_col1, hist_col2 = st.columns([5, 1])
    with hist_col1:
        st.subheader(f"📜 Prediction History ({len(st.session_state['history'])})")
    with hist_col2:
        st.write("")
        if st.button("Clear", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    hist_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(hist_df, use_container_width=True, hide_index=True, column_config={
        "Predicted demand": st.column_config.NumberColumn(format="%d bikes"),
        "Temp (°C)": st.column_config.NumberColumn(format="%.1f°C"),
        "Humidity (%)": st.column_config.NumberColumn(format="%d%%"),
    })