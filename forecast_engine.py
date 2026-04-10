import math
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="MarginCommand Forecast Lab v1.2", layout="wide")

# =========================
# CONFIG
# =========================
DAYPART_WINDOWS = {
    "Monday": {"Dinner": {"open": time(15, 0), "close": time(21, 0)}},
    "Tuesday": {"Dinner": {"open": time(15, 0), "close": time(21, 0)}},
    "Wednesday": {"Dinner": {"open": time(15, 0), "close": time(21, 0)}},
    "Thursday": {"Dinner": {"open": time(15, 0), "close": time(21, 0)}},
    "Friday": {"Dinner": {"open": time(15, 0), "close": time(22, 0)}},
    "Saturday": {
        "Lunch": {"open": time(12, 0), "close": time(15, 0)},
        "Dinner": {"open": time(15, 0), "close": time(22, 0)},
    },
    "Sunday": {
        "Lunch": {"open": time(12, 0), "close": time(15, 0)},
        "Dinner": {"open": time(15, 0), "close": time(21, 0)},
    },
}

CURVE_LIBRARY = {
    "Lunch": np.array([0.18, 0.42, 0.28, 0.12], dtype=float),
    "Dinner": {
        "Monday":    np.array([0.05, 0.09, 0.18, 0.24, 0.22, 0.16, 0.05, 0.01], dtype=float),
        "Tuesday":   np.array([0.08, 0.10, 0.20, 0.24, 0.20, 0.12, 0.05, 0.01], dtype=float),
        "Wednesday": np.array([0.07, 0.12, 0.26, 0.27, 0.19, 0.07, 0.02, 0.00], dtype=float),
        "Thursday":  np.array([0.06, 0.09, 0.18, 0.24, 0.22, 0.15, 0.05, 0.01], dtype=float),
        "Friday":    np.array([0.05, 0.11, 0.14, 0.14, 0.16, 0.18, 0.20, 0.02], dtype=float),
        "Saturday":  np.array([0.06, 0.14, 0.17, 0.20, 0.20, 0.15, 0.06, 0.02], dtype=float),
        "Sunday":    np.array([0.11, 0.16, 0.21, 0.21, 0.22, 0.06, 0.02, 0.01], dtype=float),
        "default":   np.array([0.06, 0.11, 0.19, 0.23, 0.21, 0.13, 0.06, 0.01], dtype=float),
    },
}

# This is the starting belief before live pace adjusts it.
# Stored as a decrease percentage so the UI is intuitive.
DEFAULT_DECREASE_BY_DAY = {
    "Monday": 0.32,
    "Tuesday": 0.30,
    "Wednesday": 0.26,
    "Thursday": 0.22,
    "Friday": 0.16,
    "Saturday": 0.14,
    "Sunday": 0.22,
}

PRESETS = {
    "Custom": None,
    "Slow Tuesday": {
        "day_name": "Tuesday",
        "shift": "Dinner",
        "hour12": 6,
        "minute": 0,
        "am_pm": "PM",
        "sales_so_far": 4200.0,
        "covers_so_far": 78.0,
        "total_reservations_today": 140.0,
        "normal_daily_baseline": 9200.0,
        "r365_forecast": 13800.0,
        "decrease_pct": 0.30,
        "manual_avg_per_guest": 46.0,
        "use_manual_avg": True,
    },
    "Borderline Thursday": {
        "day_name": "Thursday",
        "shift": "Dinner",
        "hour12": 6,
        "minute": 0,
        "am_pm": "PM",
        "sales_so_far": 6500.0,
        "covers_so_far": 95.0,
        "total_reservations_today": 165.0,
        "normal_daily_baseline": 12500.0,
        "r365_forecast": 14905.0,
        "decrease_pct": 0.22,
        "manual_avg_per_guest": 46.0,
        "use_manual_avg": True,
    },
    "Strong Friday Build": {
        "day_name": "Friday",
        "shift": "Dinner",
        "hour12": 6,
        "minute": 30,
        "am_pm": "PM",
        "sales_so_far": 14450.0,
        "covers_so_far": 225.0,
        "total_reservations_today": 320.0,
        "normal_daily_baseline": 21000.0,
        "r365_forecast": 23095.0,
        "decrease_pct": 0.16,
        "manual_avg_per_guest": 49.0,
        "use_manual_avg": True,
    },
}

# =========================
# HELPERS
# =========================
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def money(value: float) -> str:
    return f"${value:,.0f}"

def pct(value: float) -> str:
    return f"{value:.0%}"

def combine_today(t: time) -> datetime:
    now = datetime.now()
    return datetime(now.year, now.month, now.day, t.hour, t.minute)

def current_test_time(hour12: int, minute: int, am_pm: str) -> time:
    hour = hour12 % 12
    if am_pm.upper() == "PM":
        hour += 12
    return time(hour, minute)

def get_hours(day_name: str, shift: str) -> tuple[time, time]:
    return DAYPART_WINDOWS[day_name][shift]["open"], DAYPART_WINDOWS[day_name][shift]["close"]

def available_shifts_for_day(day_name: str) -> list[str]:
    return list(DAYPART_WINDOWS[day_name].keys())

def hours_between(start_t: time, end_t: time) -> float:
    start_dt = combine_today(start_t)
    end_dt = combine_today(end_t)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return (end_dt - start_dt).total_seconds() / 3600

def get_curve_for_shift(shift: str, hours_open: float, day_name: str = "default") -> np.ndarray:
    if shift == "Lunch":
        base = CURVE_LIBRARY["Lunch"]
    else:
        base = CURVE_LIBRARY["Dinner"].get(day_name, CURVE_LIBRARY["Dinner"]["default"])
    target_len = max(4, math.ceil(hours_open))
    x_old = np.linspace(0, 1, len(base))
    x_new = np.linspace(0, 1, target_len)
    curve = np.interp(x_new, x_old, base)
    curve = np.maximum(curve, 0.001)
    curve = curve / curve.sum()
    return curve

def expected_curve_progress(curve: np.ndarray, hours_open: float, hours_open_so_far: float) -> float:
    if hours_open <= 0:
        return 1.0
    if hours_open_so_far <= 0:
        return 0.0
    if hours_open_so_far >= hours_open:
        return 1.0

    bucket_size = hours_open / len(curve)
    progress = 0.0
    for i, weight in enumerate(curve):
        bucket_start = i * bucket_size
        bucket_end = (i + 1) * bucket_size
        if hours_open_so_far >= bucket_end:
            progress += weight
        elif hours_open_so_far > bucket_start:
            partial = (hours_open_so_far - bucket_start) / bucket_size
            progress += weight * partial
            break
        else:
            break
    return clamp(progress, 0.0, 1.0)

def live_pace_read(pace_ratio: float) -> tuple[str, str]:
    if pace_ratio < 0.85:
        return "BEHIND PACE", "red"
    if pace_ratio <= 1.05:
        return "ON PACE", "yellow"
    return "AHEAD OF PACE", "green"

def room_read_from_pace(pace_ratio: float) -> str:
    if pace_ratio < 0.90:
        return "Slow"
    if pace_ratio <= 1.05:
        return "Steady"
    return "Busy"

def live_adjusted_decrease(default_decrease: float, pace_ratio: float) -> float:
    """
    Pace below expected should increase the decrease percentage.
    Pace above expected should reduce the decrease percentage.
    Small controlled move only.
    """
    adjustment = clamp((1.0 - pace_ratio) * 0.12, -0.06, 0.06)
    return clamp(default_decrease + adjustment, 0.05, 0.45)

def confidence_label(model_only_projection: float, hybrid_projection: float, pace_ratio: float, curve_progress: float) -> str:
    if model_only_projection <= 0 or curve_progress < 0.15:
        return "LOW"
    hybrid_gap = abs(hybrid_projection - model_only_projection) / model_only_projection
    if hybrid_gap <= 0.12 and 0.85 <= pace_ratio <= 1.15:
        return "HIGH"
    if hybrid_gap <= 0.22 and 0.75 <= pace_ratio <= 1.25:
        return "MEDIUM"
    return "LOW"

def build_forecast_model(
    day_name: str,
    shift: str,
    test_t: time,
    sales_so_far: float,
    covers_so_far: float,
    total_reservations_today: float,
    normal_daily_baseline: float,
    r365_forecast: float,
    avg_per_guest: float,
    default_decrease_pct: float,
    live_decrease_pct: float,
) -> dict:
    open_t, close_t = get_hours(day_name, shift)
    open_dt = combine_today(open_t)
    close_dt = combine_today(close_t)
    test_dt = combine_today(test_t)

    if close_dt <= open_dt:
        close_dt += timedelta(days=1)
    if test_dt < open_dt:
        test_dt += timedelta(days=1) if test_dt.hour < open_t.hour else timedelta(0)

    hours_open = hours_between(open_t, close_t)
    hours_open_so_far = clamp((test_dt - open_dt).total_seconds() / 3600, 0, hours_open)
    hours_remaining = max(hours_open - hours_open_so_far, 0.0)

    curve = get_curve_for_shift(shift, hours_open, day_name)
    curve_progress = expected_curve_progress(curve, hours_open, hours_open_so_far)

    default_adjusted_baseline = normal_daily_baseline * (1 - default_decrease_pct)
    live_adjusted_baseline = normal_daily_baseline * (1 - live_decrease_pct)
    expected_sales_now = live_adjusted_baseline * curve_progress

    auto_avg_per_guest = sales_so_far / covers_so_far if covers_so_far > 0 else 0.0

    pace_projection = sales_so_far / max(curve_progress, 0.05)
    reservation_projection = total_reservations_today * max(avg_per_guest, 1)
    cover_based_projection = covers_so_far * max(avg_per_guest, 1)

    model_only_projection = (
        0.42 * pace_projection
        + 0.20 * reservation_projection
        + 0.18 * cover_based_projection
        + 0.20 * live_adjusted_baseline
    )

    floor_projection = max(
        sales_so_far * 1.02,
        sales_so_far + hours_remaining * max(avg_per_guest * 2.5, 120)
    )
    model_only_projection = max(model_only_projection, floor_projection)
    hybrid_projection = 0.70 * model_only_projection + 0.30 * r365_forecast

    pace_ratio = sales_so_far / max(expected_sales_now, 1)
    pace_read, read_color = live_pace_read(pace_ratio)
    room_read = room_read_from_pace(pace_ratio)
    confidence = confidence_label(model_only_projection, hybrid_projection, pace_ratio, curve_progress)

    sales_delta_now = sales_so_far - expected_sales_now

    return {
        "open_t": open_t,
        "close_t": close_t,
        "hours_open": hours_open,
        "hours_open_so_far": hours_open_so_far,
        "hours_remaining": hours_remaining,
        "curve_progress": curve_progress,
        "auto_avg_per_guest": auto_avg_per_guest,
        "avg_per_guest": avg_per_guest,
        "default_adjusted_baseline": default_adjusted_baseline,
        "live_adjusted_baseline": live_adjusted_baseline,
        "expected_sales_now": expected_sales_now,
        "sales_delta_now": sales_delta_now,
        "pace_projection": pace_projection,
        "reservation_projection": reservation_projection,
        "cover_based_projection": cover_based_projection,
        "model_only_projection": model_only_projection,
        "hybrid_projection": hybrid_projection,
        "pace_ratio": pace_ratio,
        "pace_read": pace_read,
        "read_color": read_color,
        "room_read": room_read,
        "confidence": confidence,
        "gap_vs_baseline": hybrid_projection - live_adjusted_baseline,
        "gap_vs_r365": hybrid_projection - r365_forecast,
        "default_decrease_pct": default_decrease_pct,
        "live_decrease_pct": live_decrease_pct,
    }

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #020817 0%, #030b1c 100%);
    color: #f8fafc;
}
.block-container {
    max-width: 1450px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}
h1,h2,h3,h4,h5,h6,p,div,span,label { color: #f8fafc !important; }
.metric {
    background: rgba(8,15,32,0.92);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 14px;
    padding: 14px 16px;
    min-height: 118px;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    color: #cbd5e1 !important;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 1.15rem;
    font-weight: 900;
    margin-bottom: 6px;
}
.metric-copy {
    font-size: 0.84rem;
    color: #cbd5e1 !important;
    line-height: 1.35;
}
.read-box {
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.10);
}
.read-green { background: linear-gradient(135deg, #166534 0%, #16a34a 100%); }
.read-yellow { background: linear-gradient(135deg, #92400e 0%, #d97706 100%); }
.read-red { background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%); }
.note-box {
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.28);
    border-radius: 14px;
    padding: 12px 14px;
    margin-top: 10px;
}
.helper-box {
    background: rgba(15,23,42,0.95);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 14px;
    padding: 12px 14px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown(
    f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:14px;">
        <div>
            <div style="font-size:2rem; font-weight:900;">MarginCommand Forecast Lab v1.2</div>
            <div style="font-size:0.84rem; color:#94a3b8;">Avg per guest logic · offseason revenue decrease logic · pace-aware read</div>
        </div>
        <div style="text-align:right; font-size:0.80rem; color:#94a3b8;">
            {datetime.now().strftime("%I:%M %p").lstrip("0")} Eastern<br>
            Forecast Engine
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
preset = PRESETS[preset_name]

if preset is None:
    day_name = "Thursday"
    shift = "Dinner"
    hour12 = 6
    minute = 0
    am_pm = "PM"
    sales_so_far = 6500.0
    covers_so_far = 95.0
    total_reservations_today = 165.0
    normal_daily_baseline = 12500.0
    r365_forecast = 14905.0
    default_decrease_pct = DEFAULT_DECREASE_BY_DAY[day_name]
    manual_avg_per_guest = 46.0
    use_manual_avg = True
else:
    day_name = preset["day_name"]
    shift = preset["shift"]
    hour12 = preset["hour12"]
    minute = preset["minute"]
    am_pm = preset["am_pm"]
    sales_so_far = preset["sales_so_far"]
    covers_so_far = preset["covers_so_far"]
    total_reservations_today = preset["total_reservations_today"]
    normal_daily_baseline = preset["normal_daily_baseline"]
    r365_forecast = preset["r365_forecast"]
    default_decrease_pct = preset["decrease_pct"]
    manual_avg_per_guest = preset["manual_avg_per_guest"]
    use_manual_avg = preset["use_manual_avg"]

left, right = st.columns([1.15, 1.0], gap="large")

with left:
    c1, c2 = st.columns(2)
    with c1:
        day_name = st.selectbox("Day", list(DAYPART_WINDOWS.keys()), index=list(DAYPART_WINDOWS.keys()).index(day_name))
    with c2:
        shifts = available_shifts_for_day(day_name)
        shift = st.selectbox("Shift", shifts, index=shifts.index(shift) if shift in shifts else 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        hour12 = st.selectbox("Hour", list(range(1, 13)), index=list(range(1, 13)).index(hour12))
    with c2:
        minute_choices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        minute = st.selectbox("Minute", minute_choices, index=minute_choices.index(minute))
    with c3:
        am_pm = st.selectbox("AM / PM", ["AM", "PM"], index=0 if am_pm == "AM" else 1)

    c1, c2 = st.columns(2)
    with c1:
        sales_so_far = st.number_input("Sales so far", min_value=0.0, value=float(sales_so_far), step=50.0)
        covers_so_far = st.number_input("Covers so far", min_value=0.0, value=float(covers_so_far), step=1.0)
        total_reservations_today = st.number_input("Total reservations today", min_value=0.0, value=float(total_reservations_today), step=1.0)
    with c2:
        normal_daily_baseline = st.number_input("Normal daily baseline", min_value=0.0, value=float(normal_daily_baseline), step=100.0)
        r365_forecast = st.number_input("R365 forecast", min_value=0.0, value=float(r365_forecast), step=100.0)

    auto_avg_per_guest_now = sales_so_far / covers_so_far if covers_so_far > 0 else 0.0
    use_manual_avg = st.checkbox("Override auto Avg per Guest", value=use_manual_avg)
    if use_manual_avg:
        avg_per_guest = st.number_input("Avg per Guest", min_value=0.0, value=float(manual_avg_per_guest), step=0.25)
    else:
        avg_per_guest = auto_avg_per_guest_now
        st.markdown(
            f"""
            <div class="helper-box">
                <div style="font-weight:800; margin-bottom:4px;">Avg per Guest is auto-filled</div>
                <div style="font-size:0.88rem;">Sales so far {money(sales_so_far)} ÷ Covers {covers_so_far:,.0f} = <strong>{money(avg_per_guest)}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # DSS weekday starting point
    default_decrease_pct = DEFAULT_DECREASE_BY_DAY[day_name]

    test_t_for_preview = current_test_time(hour12, minute, am_pm)
    open_t_preview, close_t_preview = get_hours(day_name, shift)
    hours_open_preview = hours_between(open_t_preview, close_t_preview)
    open_dt_preview = combine_today(open_t_preview)
    test_dt_preview = combine_today(test_t_for_preview)
    if test_dt_preview < open_dt_preview:
        test_dt_preview += timedelta(days=1) if test_dt_preview.hour < open_t_preview.hour else timedelta(0)

    hours_open_so_far_preview = clamp((test_dt_preview - open_dt_preview).total_seconds() / 3600, 0, hours_open_preview)
    curve_preview = get_curve_for_shift(shift, hours_open_preview, day_name)
    curve_progress_preview = expected_curve_progress(curve_preview, hours_open_preview, hours_open_so_far_preview)

    default_adjusted_baseline_preview = normal_daily_baseline * (1 - default_decrease_pct)
    expected_sales_now_preview = default_adjusted_baseline_preview * curve_progress_preview
    preview_pace_ratio = sales_so_far / max(expected_sales_now_preview, 1)
    auto_live_decrease = live_adjusted_decrease(default_decrease_pct, preview_pace_ratio)

    use_auto_decrease = st.checkbox("Auto adjust offseason decrease from pace", value=True)
    if use_auto_decrease:
        live_decrease_pct = auto_live_decrease
        st.markdown(
            f"""
            <div class="helper-box">
                <div style="font-weight:800; margin-bottom:4px;">Offseason Revenue Decrease is auto-adjusted</div>
                <div style="font-size:0.88rem;">
                    Default by day: <strong>{pct(default_decrease_pct)}</strong><br>
                    Live adjusted from pace: <strong>{pct(live_decrease_pct)}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        live_decrease_pct = st.slider(
            "Offseason Revenue Decrease %",
            min_value=0.05,
            max_value=0.45,
            value=float(round(default_decrease_pct, 2)),
            step=0.01,
        )

    st.markdown(
        """
        <div class="note-box">
            <div style="font-weight:800; margin-bottom:4px;">How to think about this input</div>
            <div style="font-size:0.86rem; line-height:1.5;">
                This is a revenue decrease percentage, not a raw multiplier.
                Higher percentage = bigger offseason drop.
                Lower percentage = lighter drop.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

test_t = current_test_time(hour12, minute, am_pm)
model = build_forecast_model(
    day_name=day_name,
    shift=shift,
    test_t=test_t,
    sales_so_far=sales_so_far,
    covers_so_far=covers_so_far,
    total_reservations_today=total_reservations_today,
    normal_daily_baseline=normal_daily_baseline,
    r365_forecast=r365_forecast,
    avg_per_guest=avg_per_guest,
    default_decrease_pct=default_decrease_pct,
    live_decrease_pct=live_decrease_pct,
)

read_class = {"green": "read-green", "yellow": "read-yellow", "red": "read-red"}[model["read_color"]]

with right:
    st.markdown(
        f"""
        <div class="read-box {read_class}">
            <div style="font-size:0.78rem; font-weight:800; text-transform:uppercase; letter-spacing:0.06em;">Live Pace Read</div>
            <div style="font-size:1.55rem; font-weight:900; margin-top:6px;">{model['pace_read']}</div>
            <div style="font-size:0.92rem; margin-top:8px;">
                Pace ratio: <strong>{model['pace_ratio']:.2f}x</strong><br>
                Room is reading: <strong>{model['room_read']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""<div class="metric"><div class="metric-label">Hybrid projection</div>
            <div class="metric-value">{money(model['hybrid_projection'])}</div>
            <div class="metric-copy">70% local model + 30% R365</div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric"><div class="metric-label">Confidence</div>
            <div class="metric-value">{model['confidence']}</div>
            <div class="metric-copy">Based on curve progress, pace alignment, and model gap</div></div>""",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""<div class="metric"><div class="metric-label">Live adjusted baseline</div>
            <div class="metric-value">{money(model['live_adjusted_baseline'])}</div>
            <div class="metric-copy">Normal baseline after offseason revenue decrease</div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric"><div class="metric-label">Expected sales now</div>
            <div class="metric-value">{money(model['expected_sales_now'])}</div>
            <div class="metric-copy">Where sales should be at this exact point in the shift</div></div>""",
            unsafe_allow_html=True,
        )

    delta_copy = "Ahead of expected" if model["sales_delta_now"] >= 0 else "Behind expected"
    st.markdown(
        f"""
        <div class="note-box">
            <div style="font-weight:800; margin-bottom:4px;">Why the read looks this way</div>
            <div style="font-size:0.88rem; line-height:1.5;">
                Actual sales: <strong>{money(sales_so_far)}</strong><br>
                Expected sales now: <strong>{money(model['expected_sales_now'])}</strong><br>
                Delta: <strong>{money(model['sales_delta_now'])}</strong> · {delta_copy}<br>
                Default DSS decrease: <strong>{pct(model['default_decrease_pct'])}</strong><br>
                Live adjusted decrease: <strong>{pct(model['live_decrease_pct'])}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border:none; border-top:1px solid rgba(148,163,184,0.15); margin:20px 0;'>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
for col, label, value, copy in [
    (m1, "Avg per Guest", money(model["avg_per_guest"]), "This is the revenue-per-guest input used by the model"),
    (m2, "Gap vs adjusted baseline", money(model["gap_vs_baseline"]), "How far hybrid projection sits above or below the live adjusted baseline"),
    (m3, "Gap vs R365", money(model["gap_vs_r365"]), "How far hybrid projection sits above or below R365"),
    (m4, "Curve progress captured", f"{model['curve_progress']:.1%}", "Demand-weighted shift progress, not just clock time"),
]:
    with col:
        st.markdown(
            f"""<div class="metric"><div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div><div class="metric-copy">{copy}</div></div>""",
            unsafe_allow_html=True,
        )

layers = pd.DataFrame(
    {
        "Layer": [
            "Pace projection",
            "Reservation projection",
            "Cover based projection",
            "Default adjusted baseline",
            "Live adjusted baseline",
            "Model only projection",
            "Hybrid final projection",
        ],
        "Value": [
            model["pace_projection"],
            model["reservation_projection"],
            model["cover_based_projection"],
            model["default_adjusted_baseline"],
            model["live_adjusted_baseline"],
            model["model_only_projection"],
            model["hybrid_projection"],
        ],
    }
)
layers["Value"] = layers["Value"].map(money)

st.markdown("### Forecast build")
st.dataframe(layers, use_container_width=True, hide_index=True)

open_str = model["open_t"].strftime("%I:%M %p").lstrip("0")
close_str = model["close_t"].strftime("%I:%M %p").lstrip("0")
st.markdown(
    f"""
    <div class="note-box">
        <div style="font-weight:800; margin-bottom:4px;">How to use v1.2</div>
        <div style="font-size:0.86rem; line-height:1.5;">
            This model uses <strong>Avg per Guest</strong>, not ticket average or table average.
            Start with the DSS weekday revenue decrease. That is the pre-shift seasonal drop.
            Then let live pace adjust that decrease slightly if the day is clearly running softer or stronger than expected.
            <br><br>
            Open: {open_str} &nbsp;·&nbsp; Close: {close_str} &nbsp;·&nbsp;
            Hours open so far: {model['hours_open_so_far']:.2f} &nbsp;·&nbsp;
            Hours remaining: {model['hours_remaining']:.2f}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
