import math
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='MarginCommand Forecast Lab', layout='wide')

DAYPART_WINDOWS = {
    'Monday': {'Dinner': {'open': time(15, 0), 'close': time(21, 0)}},
    'Tuesday': {'Dinner': {'open': time(15, 0), 'close': time(21, 0)}},
    'Wednesday': {'Dinner': {'open': time(15, 0), 'close': time(21, 0)}},
    'Thursday': {'Dinner': {'open': time(15, 0), 'close': time(21, 0)}},
    'Friday': {'Dinner': {'open': time(15, 0), 'close': time(22, 0)}},
    'Saturday': {
        'Lunch': {'open': time(12, 0), 'close': time(15, 0)},
        'Dinner': {'open': time(15, 0), 'close': time(22, 0)},
    },
    'Sunday': {
        'Lunch': {'open': time(12, 0), 'close': time(15, 0)},
        'Dinner': {'open': time(15, 0), 'close': time(21, 0)},
    },
}

CURVE_LIBRARY = {
    'Lunch': np.array([0.18, 0.42, 0.28, 0.12], dtype=float),
    'Dinner': {
        'Monday':    np.array([0.05, 0.09, 0.18, 0.24, 0.22, 0.16, 0.05, 0.01], dtype=float),
        'Tuesday':   np.array([0.08, 0.10, 0.20, 0.24, 0.20, 0.12, 0.05, 0.01], dtype=float),
        'Wednesday': np.array([0.07, 0.12, 0.26, 0.27, 0.19, 0.07, 0.02, 0.00], dtype=float),
        'Thursday':  np.array([0.06, 0.09, 0.18, 0.24, 0.22, 0.15, 0.05, 0.01], dtype=float),
        'Friday':    np.array([0.05, 0.11, 0.14, 0.14, 0.16, 0.18, 0.20, 0.02], dtype=float),
        'Saturday':  np.array([0.06, 0.14, 0.17, 0.20, 0.20, 0.15, 0.06, 0.02], dtype=float),
        'Sunday':    np.array([0.11, 0.16, 0.21, 0.21, 0.22, 0.06, 0.02, 0.01], dtype=float),
        'default':   np.array([0.06, 0.11, 0.19, 0.23, 0.21, 0.13, 0.06, 0.01], dtype=float),
    },
}

OFF_SEASON_FACTORS = {
    'Monday': 0.68,
    'Tuesday': 0.68,
    'Wednesday': 0.70,
    'Thursday': 0.73,
    'Friday': 0.80,
    'Saturday': 0.84,
    'Sunday': 0.76,
}

APRIL_BASELINES = {
    'Monday': 9000.0,
    'Tuesday': 9200.0,
    'Wednesday': 11000.0,
    'Thursday': 12500.0,
    'Friday': 21000.0,
    'Saturday': 26000.0,
    'Sunday': 20000.0,
}

PRESETS = {
    'Custom': None,
    'Soft Thursday Off Season': {
        'day_name': 'Thursday',
        'shift': 'Dinner',
        'hour12': 6,
        'minute': 10,
        'am_pm': 'PM',
        'sales_so_far': 6450.0,
        'average_check': 46.0,
        'covers_so_far': 96.0,
        'total_reservations_today': 165.0,
        'daily_forecast_baseline': 12500.0,
        'r365_forecast': 14905.0,
        'dining_room_feel': 2,
        'off_season_factor': 0.73,
    },
    'Strong Friday Build': {
        'day_name': 'Friday',
        'shift': 'Dinner',
        'hour12': 6,
        'minute': 30,
        'am_pm': 'PM',
        'sales_so_far': 14450.0,
        'average_check': 49.0,
        'covers_so_far': 225.0,
        'total_reservations_today': 320.0,
        'daily_forecast_baseline': 21000.0,
        'r365_forecast': 23095.0,
        'dining_room_feel': 4,
        'off_season_factor': 0.80,
    },
    'Saturday Holding': {
        'day_name': 'Saturday',
        'shift': 'Dinner',
        'hour12': 7,
        'minute': 0,
        'am_pm': 'PM',
        'sales_so_far': 17500.0,
        'average_check': 48.0,
        'covers_so_far': 258.0,
        'total_reservations_today': 335.0,
        'daily_forecast_baseline': 26000.0,
        'r365_forecast': 26818.0,
        'dining_room_feel': 3,
        'off_season_factor': 0.84,
    },
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def money(value: float) -> str:
    return f'${value:,.0f}' if abs(value) >= 100 else f'${value:,.2f}'


def combine_today(t: time) -> datetime:
    now = datetime.now()
    return datetime(now.year, now.month, now.day, t.hour, t.minute)


def hours_between(start_t: time, end_t: time) -> float:
    start_dt = combine_today(start_t)
    end_dt = combine_today(end_t)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return (end_dt - start_dt).total_seconds() / 3600


def current_test_time(hour12: int, minute: int, am_pm: str) -> time:
    hour = hour12 % 12
    if am_pm.upper() == 'PM':
        hour += 12
    return time(hour, minute)


def get_curve_for_shift(shift: str, hours_open: float, day_name: str) -> np.ndarray:
    if shift == 'Lunch':
        base = CURVE_LIBRARY['Lunch']
    else:
        dinner_curves = CURVE_LIBRARY['Dinner']
        base = dinner_curves.get(day_name, dinner_curves['default'])
    target_len = max(4, math.ceil(hours_open))
    x_old = np.linspace(0, 1, len(base))
    x_new = np.linspace(0, 1, target_len)
    curve = np.interp(x_new, x_old, base)
    curve = np.maximum(curve, 0.001)
    return curve / curve.sum()


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


def build_forecast_model(
    day_name: str,
    shift: str,
    test_t: time,
    sales_so_far: float,
    average_check: float,
    covers_so_far: float,
    total_reservations_today: float,
    daily_forecast_baseline: float,
    r365_forecast: float,
    dining_room_feel: int,
    off_season_factor: float,
) -> dict:
    open_t = DAYPART_WINDOWS[day_name][shift]['open']
    close_t = DAYPART_WINDOWS[day_name][shift]['close']

    open_dt = combine_today(open_t)
    close_dt = combine_today(close_t)
    if close_dt <= open_dt:
        close_dt += timedelta(days=1)

    test_dt = combine_today(test_t)
    if test_dt < open_dt:
        test_dt += timedelta(days=1) if test_dt.hour < open_t.hour else timedelta(0)

    hours_open = hours_between(open_t, close_t)
    hours_open_so_far = clamp((test_dt - open_dt).total_seconds() / 3600, 0, hours_open)
    hours_remaining = max(hours_open - hours_open_so_far, 0)

    curve = get_curve_for_shift(shift, hours_open, day_name)
    curve_progress = expected_curve_progress(curve, hours_open, hours_open_so_far)

    pace_projection = sales_so_far / max(curve_progress, 0.05)
    cover_based_projection = max(covers_so_far, 1) * max(average_check, 1)
    reservation_projection = max(total_reservations_today, covers_so_far) * max(average_check, 1)
    floor_projection = sales_so_far + max(hours_remaining, 0) * max(average_check * 2.5, 120)

    feel_adjust = {1: 0.94, 2: 0.98, 3: 1.00, 4: 1.04, 5: 1.08}.get(dining_room_feel, 1.0)

    off_season_baseline = daily_forecast_baseline * off_season_factor

    model_projection = (
        0.45 * pace_projection
        + 0.20 * reservation_projection
        + 0.15 * cover_based_projection
        + 0.10 * off_season_baseline
        + 0.10 * max(sales_so_far, 1)
    ) * feel_adjust

    model_projection = max(model_projection, floor_projection, sales_so_far)
    hybrid_projection = 0.70 * model_projection + 0.30 * r365_forecast
    hybrid_projection = max(hybrid_projection, sales_so_far)

    forecast_gap = hybrid_projection - off_season_baseline
    r365_gap = hybrid_projection - r365_forecast

    if forecast_gap <= -2500:
        trend_flag = 'UNDER DAY'
        trend_copy = 'Running below your off season day target. Protect margin first.'
    elif forecast_gap >= 2500:
        trend_flag = 'OVER DAY'
        trend_copy = 'Tracking above the off season target. Hold longer if labor still fits.'
    else:
        trend_flag = 'ON TRACK'
        trend_copy = 'Tracking close to the off season baseline for this day.'

    if r365_gap <= -1500:
        compare_copy = 'Below R365. Your local model is calling for a softer finish.'
    elif r365_gap >= 1500:
        compare_copy = 'Above R365. Live pace and local conditions support a stronger finish.'
    else:
        compare_copy = 'Very close to R365 right now.'

    confidence = 'LOW'
    if hours_open_so_far >= hours_open * 0.20 or sales_so_far >= 1500:
        confidence = 'MEDIUM'
    if hours_open_so_far >= hours_open * 0.35 and covers_so_far >= 40:
        confidence = 'HIGH'

    return {
        'open_t': open_t,
        'close_t': close_t,
        'hours_open': hours_open,
        'hours_open_so_far': hours_open_so_far,
        'hours_remaining': hours_remaining,
        'curve_progress': curve_progress,
        'pace_projection': pace_projection,
        'reservation_projection': reservation_projection,
        'cover_based_projection': cover_based_projection,
        'off_season_baseline': off_season_baseline,
        'model_projection': model_projection,
        'hybrid_projection': hybrid_projection,
        'forecast_gap': forecast_gap,
        'r365_gap': r365_gap,
        'trend_flag': trend_flag,
        'trend_copy': trend_copy,
        'compare_copy': compare_copy,
        'confidence': confidence,
    }


st.title('MarginCommand Forecast Lab')
st.caption('Separate local model for daily forecast testing before folding into the main app.')

preset_name = st.selectbox('Preset', list(PRESETS.keys()))
preset = PRESETS[preset_name]

if preset is None:
    default_day = 'Thursday'
    default_shift = 'Dinner'
    default_hour12 = 6
    default_minute = 0
    default_am_pm = 'PM'
    default_sales = 6500.0
    default_avg_check = 46.0
    default_covers = 95.0
    default_res = 165.0
    default_daily = APRIL_BASELINES[default_day]
    default_r365 = 14905.0
    default_feel = 3
    default_off = OFF_SEASON_FACTORS[default_day]
else:
    default_day = preset['day_name']
    default_shift = preset['shift']
    default_hour12 = preset['hour12']
    default_minute = preset['minute']
    default_am_pm = preset['am_pm']
    default_sales = preset['sales_so_far']
    default_avg_check = preset['average_check']
    default_covers = preset['covers_so_far']
    default_res = preset['total_reservations_today']
    default_daily = preset['daily_forecast_baseline']
    default_r365 = preset['r365_forecast']
    default_feel = preset['dining_room_feel']
    default_off = preset['off_season_factor']

with st.form('forecast_lab_form'):
    c1, c2, c3 = st.columns(3)
    with c1:
        day_name = st.selectbox('Day', list(DAYPART_WINDOWS.keys()), index=list(DAYPART_WINDOWS.keys()).index(default_day))
    with c2:
        shift = st.selectbox('Shift', list(DAYPART_WINDOWS[day_name].keys()), index=list(DAYPART_WINDOWS[day_name].keys()).index(default_shift) if default_shift in DAYPART_WINDOWS[day_name] else 0)
    with c3:
        dining_room_feel = st.slider('Dining room feel', 1, 5, default_feel)

    c1, c2, c3 = st.columns(3)
    with c1:
        hour12 = st.selectbox('Hour', list(range(1, 13)), index=list(range(1, 13)).index(default_hour12))
    with c2:
        minute = st.selectbox('Minute', [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], index=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55].index(default_minute))
    with c3:
        am_pm = st.selectbox('AM / PM', ['AM', 'PM'], index=0 if default_am_pm == 'AM' else 1)

    c1, c2 = st.columns(2)
    with c1:
        sales_so_far = st.number_input('Sales so far', min_value=0.0, value=float(default_sales), step=50.0)
        average_check = st.number_input('Average check', min_value=0.0, value=float(default_avg_check), step=0.25)
        covers_so_far = st.number_input('Covers so far', min_value=0.0, value=float(default_covers), step=1.0)
    with c2:
        total_reservations_today = st.number_input('Total reservations today', min_value=0.0, value=float(default_res), step=1.0)
        daily_forecast_baseline = st.number_input('Your daily baseline', min_value=0.0, value=float(default_daily), step=100.0)
        r365_forecast = st.number_input('R365 forecast', min_value=0.0, value=float(default_r365), step=100.0)

    off_season_factor = st.slider('Off season factor', min_value=0.50, max_value=1.00, value=float(default_off), step=0.01)
    submitted = st.form_submit_button('Run forecast model', use_container_width=True)

if submitted or True:
    test_t = current_test_time(hour12, minute, am_pm)
    result = build_forecast_model(
        day_name=day_name,
        shift=shift,
        test_t=test_t,
        sales_so_far=sales_so_far,
        average_check=average_check,
        covers_so_far=covers_so_far,
        total_reservations_today=total_reservations_today,
        daily_forecast_baseline=daily_forecast_baseline,
        r365_forecast=r365_forecast,
        dining_room_feel=dining_room_feel,
        off_season_factor=off_season_factor,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Hybrid projection', money(result['hybrid_projection']))
    c2.metric('Off season baseline', money(result['off_season_baseline']))
    c3.metric('Gap vs baseline', money(result['forecast_gap']))
    c4.metric('Gap vs R365', money(result['r365_gap']))

    st.subheader(f"{result['trend_flag']} · Confidence {result['confidence']}")
    st.write(result['trend_copy'])
    st.write(result['compare_copy'])

    breakdown = pd.DataFrame([
        ['Pace projection', result['pace_projection']],
        ['Reservation projection', result['reservation_projection']],
        ['Cover based projection', result['cover_based_projection']],
        ['Your off season baseline', result['off_season_baseline']],
        ['Model only projection', result['model_projection']],
        ['Hybrid final projection', result['hybrid_projection']],
    ], columns=['Layer', 'Value'])
    breakdown['Value'] = breakdown['Value'].map(money)
    st.dataframe(breakdown, use_container_width=True, hide_index=True)

    st.markdown('### Local test notes')
    st.markdown(
        '\n'.join([
            f"Open: {result['open_t'].strftime('%I:%M %p').lstrip('0')}",
            f"Close: {result['close_t'].strftime('%I:%M %p').lstrip('0')}",
            f"Hours open so far: {result['hours_open_so_far']:.2f}",
            f"Hours remaining: {result['hours_remaining']:.2f}",
            f"Curve progress captured: {result['curve_progress']:.1%}",
        ])
    )

    st.info('Use this lab to test the forecast layer by itself. Once the projections feel right, fold the same math into the main MarginCommand app.')
