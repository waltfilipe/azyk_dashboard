import re
import os
import math
from pathlib import Path
from io import BytesIO
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap
import plotly.graph_objects as go

# PAGE CONFIG
st.set_page_config(layout="wide", page_title="Hudson Cicala — Dashboard")

# OPTIONAL DOCX IMPORT
DOCX_AVAILABLE = True
try:
    from docx import Document
except Exception:
    DOCX_AVAILABLE = False

# STYLE
st.markdown("""
""", unsafe_allow_html=True)

# CONSTANTS
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67
GOAL_X = 120.0
GOAL_Y = 40.0
FIG_W, FIG_H = 7.0, 4.7
FIG_DPI = 180
COLOR_SUCCESS = "#c8c8c8"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_FAIL = "#E07070"
ALPHA_SUCCESS = 0.07
COLOR_CROSS_WON = "#10b981"
COLOR_CROSS_LOST = "#b91c1c"
C_BLUE = "#2F80ED"
C_BLUE_DARK = "#1a56db"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_PURPLE_LIGHT = "#a78bfa"
C_BLUE_PASTEL = "#5b9bd5"
C_GREEN_PASTEL = "#70ad47"
C_AMBER_PASTEL = "#d4a843"
CMAP_TOP10 = LinearSegmentedColormap.from_list("top10", ["#fef08a", "#f97316", "#b91c1c"])
NORM_TOP10 = Normalize(vmin=0.05, vmax=0.40)
NX_XT, NY_XT = 16, 12
D_REF, D_SCALE, BONUS_CAP = 10.0, 20.0, 0.60
LATERAL_MIN_DIST = 12.0
PENALTY_AREA_X = 18.0
FUNNEL_X_EXTEND = 33.0
PENALTY_AREA_Y_MIN = 18.0
PENALTY_AREA_Y_MAX = 62.0

def _hex_to_rgba(hex_color, alpha=1.0):
    if hex_color.startswith('#'):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'
    return hex_color

def get_lane(y):
    if y >= LANE_LEFT_MIN:
        return "left"
    elif y < LANE_RIGHT_MAX:
        return "right"
    return "center"

def distance_to_goal(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)

def is_progressive_pass(x_start, y_start, x_end, y_end):
    if x_start < 35:
        return False
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    if start_dist == 0:
        return False
    return ((start_dist - end_dist) / start_dist) >= 0.25

def classify_pass_direction(x_start, y_start, x_end, y_end):
    dx = x_end - x_start
    dy = y_end - y_start
    dist = np.sqrt(dx ** 2 + dy ** 2)
    angle_deg = np.degrees(np.arctan2(abs(dy), dx))
    if angle_deg <= 45.0:
        return "forward"
    if angle_deg >= 135.0:
        return "backward"
    if dist > LATERAL_MIN_DIST:
        return "lateral_right" if dy > 0 else "lateral_left"
    return "forward" if dx >= 0 else "backward"

def distance_bonus(distance):
    excess = np.maximum(0.0, np.asarray(distance, dtype=float) - D_REF)
    return np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))

@st.cache_data(show_spinner=False)
def compute_xt_grid(NX=16, NY=12, sub=24):
    ncols_hr = NX * sub
    nrows_hr = NY * sub
    xe = np.linspace(0, FIELD_X, ncols_hr + 1)
    ye = np.linspace(0, FIELD_Y, nrows_hr + 1)
    xc = (xe[:-1] + xe[1:]) / 2
    yc_arr = (ye[:-1] + ye[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc_arr)
    xp = 0.01 + (Xc / FIELD_X) * 0.99
    yc = 1.0 - np.abs((Yc / FIELD_Y) - 0.5) * 2.0
    base = xp * (0.8 + 0.2 * yc)
    base = (base - base.min()) / (base.max() - base.min() + 1e-12)
    XT = base.copy()
    XT = (XT - XT.min()) / (XT.max() - XT.min() + 1e-12)
    XTc = np.zeros((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            XTc[iy, ix] = XT[iy * sub:(iy + 1) * sub, ix * sub:(ix + 1) * sub].mean()
    XTc = (XTc - XTc.min()) / (XTc.max() - XTc.min() + 1e-12)
    return XTc

XT_GRID = compute_xt_grid()

def xt_value(x, y):
    ix = int(np.clip((x / FIELD_X) * NX_XT, 0, NX_XT - 1))
    iy = int(np.clip((y / FIELD_Y) * NY_XT, 0, NY_XT - 1))
    return float(XT_GRID[iy, ix])

def is_in_funnel_zone(x, y):
    return x <= FUNNEL_X_EXTEND and PENALTY_AREA_Y_MIN <= y <= PENALTY_AREA_Y_MAX

# BASE PASSES
BASE_MATCHES_DATA = {
    "Real Salt Lake (05-10)": [
        ("PASS_WON", 115.52, 72.91, 109.54, 73.08, None),
        ("PASS_WON", 108.21, 53.46, 108.04, 39.67, None),
        ("PASS_WON", 103.05, 66.10, 96.24, 50.81, None),
        ("PASS_WON", 98.07, 73.08, 90.59, 61.94, None),
        ("PASS_WON", 92.25, 67.10, 74.63, 53.13, None),
        ("PASS_WON", 89.59, 74.91, 80.28, 60.61, None),
        ("PASS_WON", 69.31, 53.46, 72.97, 64.94, None),
        ("PASS_WON", 63.66, 68.76, 75.13, 49.14, None),
        ("PASS_WON", 67.65, 63.44, 67.65, 51.14, None),
        ("PASS_WON", 73.30, 74.91, 52.35, 66.60, None),
        ("PASS_WON", 55.35, 64.44, 49.53, 48.64, None),
        ("PASS_WON", 53.18, 51.97, 42.21, 41.16, None),
        ("PASS_WON", 42.71, 61.28, 41.55, 38.50, None),
        ("PASS_WON", 19.77, 58.62, 42.38, 58.12, None),
        ("PASS_WON", 23.93, 70.92, 8.30, 52.14, None),
        ("PASS_WON", 66.15, 58.95, 94.24, 68.76, None),
        ("PASS_WON", 97.90, 69.59, 88.76, 66.76, None),
        ("PASS_WON", 74.30, 65.77, 62.99, 53.80, None),
        ("PASS_WON", 69.48, 69.59, 60.66, 64.44, None),
        ("PASS_WON", 62.99, 72.91, 45.70, 62.77, None),
        ("PASS_WON", 51.19, 69.59, 51.02, 54.63, None),
        ("PASS_WON", 46.54, 74.58, 38.56, 62.77, None),
        ("PASS_WON", 39.39, 66.10, 49.36, 60.78, None),
        ("PASS_WON", 42.71, 67.59, 45.21, 57.62, None),
        ("PASS_WON", 47.86, 61.28, 35.23, 50.14, None),
        ("CROSS_WON", 89.92, 70.42, 103.22, 30.36, None),
        ("PASS_LOST", 44.21, 63.27, 69.48, 19.72, None),
        ("PASS_LOST", 49.69, 71.58, 55.51, 60.61, None),
        ("PASS_LOST", 36.06, 55.96, 23.26, 56.29, None),
        ("PASS_LOST", 18.11, 74.08, 18.94, 60.61, None),
        ("CROSS_LOST", 114.86, 70.92, 110.70, 40.17, None),
        ("CROSS_LOST", 105.88, 72.42, 111.53, 62.61, None),
    ],
    "Vancouver Whitecaps (05-15)": [
        ("PASS_WON", 106.55, 57.45, 118.02, 57.12, None),
        ("PASS_WON", 85.60, 71.92, 84.27, 56.62, None),
        ("PASS_WON", 97.57, 65.10, 88.09, 54.13, None),
        ("PASS_WON", 77.62, 63.61, 90.59, 50.14, None),
        ("PASS_WON", 66.32, 56.62, 79.95, 46.15, None),
        ("PASS_WON", 83.61, 74.58, 103.22, 67.76, None),
        ("PASS_WON", 74.96, 59.28, 95.57, 75.57, None),
        ("PASS_WON", 84.94, 60.45, 70.47, 53.80, None),
        ("PASS_WON", 70.81, 66.76, 64.65, 51.80, None),
        ("PASS_WON", 58.17, 64.27, 59.67, 55.96, None),
        ("PASS_WON", 56.01, 68.76, 58.17, 55.46, None),
        ("PASS_WON", 56.34, 70.59, 54.51, 55.79, None),
        ("PASS_WON", 55.35, 73.91, 81.94, 64.27, None),
        ("PASS_WON", 47.70, 63.61, 83.77, 78.23, None),
        ("PASS_WON", 48.70, 68.59, 49.03, 54.13, None),
        ("PASS_WON", 27.25, 69.92, 46.70, 78.07, None),
        ("PASS_WON", 36.39, 69.09, 59.34, 61.28, None),
        ("PASS_WON", 42.71, 69.76, 17.28, 68.26, None),
        ("PASS_WON", 25.26, 67.26, 32.24, 53.63, None),
        ("PASS_LOST", 38.89, 63.27, 48.03, 62.61, None),
        ("PASS_LOST", 51.85, 66.26, 101.39, 66.93, None),
        ("PASS_LOST", 62.99, 63.61, 90.59, 78.90, None),
    ],
    "LAFC (05-20)": [
        ("PASS_WON", 97.40, 56.29, 117.68, 56.12, None),
        ("PASS_WON", 99.40, 75.91, 93.58, 66.76, None),
        ("PASS_WON", 82.44, 61.78, 74.13, 55.29, None),
        ("PASS_WON", 75.79, 59.28, 87.76, 72.08, None),
        ("PASS_WON", 63.66, 68.76, 72.63, 55.63, None),
        ("PASS_WON", 46.37, 57.29, 66.82, 77.57, None),
        ("PASS_WON", 44.54, 56.46, 38.06, 54.63, None),
        ("PASS_WON", 27.25, 60.45, 34.57, 49.31, None),
        ("PASS_WON", 37.23, 67.10, 22.43, 62.61, None),
        ("PASS_WON", 33.40, 70.75, 24.92, 61.28, None),
        ("PASS_WON", 25.76, 71.42, 18.28, 66.76, None),
        ("PASS_WON", 20.27, 76.24, 35.90, 73.91, None),
        ("PASS_WON", 13.12, 53.63, 11.29, 24.21, None),
        ("PASS_LOST", 112.20, 63.27, 103.89, 56.29, None),
        ("CROSS_LOST", 99.90, 61.94, 107.54, 36.68, None),
    ],
}

# DEFENSIVE ACTIONS
DEFENSIVE_MATCHES_DATA = {
    "Real Salt Lake (05-10)": [
        ("DUEL_WON", 109.70, 68.92),
        ("DUEL_WON", 68.64, 55.79),
        ("DUEL_WON", 19.44, 78.40),
        ("DUEL_WON", 28.08, 69.42),
        ("DUEL_LOST", 2.65, 73.75),
        ("DUEL_LOST", 17.28, 64.94),
        ("DUEL_LOST", 29.25, 58.62),
        ("INTERCEPTION", 27.75, 56.96),
        ("INTERCEPTION", 24.26, 59.95),
        ("INTERCEPTION", 10.96, 51.64),
    ],
    "Vancouver Whitecaps (05-15)": [
        ("DUEL_WON", 49.19, 64.77),
        ("INTERCEPTION", 75.96, 63.61),
        ("INTERCEPTION", 27.25, 50.97),
        ("INTERCEPTION", 35.23, 66.76),
    ],
    "LAFC (05-20)": [
        ("DUEL_WON", 55.35, 70.59),
        ("DUEL_WON", 31.74, 65.77),
        ("DUEL_WON", 21.93, 64.27),
        ("DUEL_LOST", 24.92, 53.13),
        ("DUEL_LOST", 13.29, 68.09),
        ("DUEL_LOST", 5.48, 72.91),
        ("DUEL_LOST", 1.15, 69.92),
        ("INTERCEPTION", 41.22, 60.45),
        ("INTERCEPTION", 25.42, 64.27),
        ("INTERCEPTION", 25.59, 67.76),
    ],
}

def get_match_minutes(match_name: str) -> float:
    if match_name == "All Matches":
        total = 0.0
        for k in dfs_by_match:
            total += get_match_minutes(k)
        return total
    name_lower = match_name.lower()
    if "salt lake" in name_lower:
        return 85.0
    if "vancouver" in name_lower:
        return 45.0
    if "lafc" in name_lower or "los angeles" in name_lower:
        return 82.0
    return 90.0

# DATA LOADING
combined_matches_data = {}
for k, v in BASE_MATCHES_DATA.items():
    combined_matches_data[k] = v

if len(combined_matches_data) == 0:
    st.error("Could not load data.")
    st.stop()

dfs_by_match = {}
for match_name, events in combined_matches_data.items():
    dfm = pd.DataFrame(events, columns=["type", "x_start", "y_start", "x_end", "y_end", "video"])
    dfm["match"] = match_name
    dfm["number"] = np.arange(1, len(dfm) + 1)
    dfm["is_won"] = dfm["type"].str.contains("WON", case=False) | (dfm["type"] == "CROSS_WON")
    dfm["is_cross"] = dfm["type"].str.contains("CROSS", case=False)
    dfm["progressive"] = dfm.apply(
        lambda r: r["is_won"] and not r["is_cross"] and is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
        axis=1
    )
    dfm["direction"] = dfm.apply(
        lambda r: classify_pass_direction(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
        axis=1
    )
    dfm["is_forward"] = dfm["direction"] == "forward"
    dfm["is_backward"] = dfm["direction"] == "backward"
    dfm["is_lateral"] = dfm["direction"].isin(["lateral_left", "lateral_right"])
    dfm["pass_distance"] = np.sqrt((dfm["x_end"] - dfm["x_start"]) ** 2 + (dfm["y_end"] - dfm["y_start"]) ** 2)
    dfm["xt_start"] = dfm.apply(lambda r: xt_value(r["x_start"], r["y_start"]), axis=1)
    dfm["xt_end"] = dfm.apply(lambda r: xt_value(r["x_end"], r["y_end"]), axis=1)
    dfm["delta_xt"] = np.where(dfm["is_won"], dfm["xt_end"] - dfm["xt_start"], 0.0)
    dfm["dist_bonus"] = distance_bonus(dfm["pass_distance"].values)
    dfm["delta_xt_adj"] = np.where(dfm["is_won"], dfm["delta_xt"] * (1.0 + dfm["dist_bonus"]), 0.0)
    dfs_by_match[match_name] = dfm

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)

defensive_dfs_by_match = {}
for match_name, events in DEFENSIVE_MATCHES_DATA.items():
    df_def = pd.DataFrame(events, columns=["type", "x", "y"])
    df_def["match"] = match_name
    df_def["is_attacking_half"] = df_def["x"] >= FIELD_X / 2
    df_def["is_duel_won"] = df_def["type"] == "DUEL_WON"
    df_def["is_duel_lost"] = df_def["type"] == "DUEL_LOST"
    df_def["is_duel"] = df_def["is_duel_won"] | df_def["is_duel_lost"]
    df_def["is_interception"] = df_def["type"] == "INTERCEPTION"
    df_def["in_funnel"] = df_def.apply(lambda r: is_in_funnel_zone(r["x"], r["y"]), axis=1)
    defensive_dfs_by_match[match_name] = df_def

ACTIVE_PASS_MATCHES = list(dfs_by_match.keys())
ACTIVE_DEF_MATCHES = list(defensive_dfs_by_match.keys())

def compute_stats(df: pd.DataFrame, match_name: str) -> dict:
    total = len(df)
    mins = get_match_minutes(match_name)
    p90_factor = 90.0 / mins if mins > 0 else 1.0
    if total == 0:
        return {
            "total_passes": 0, "successful_passes": 0, "unsuccessful_passes": 0,
            "accuracy_pct": 0.0, "progressive_attempted": 0, "progressive_successful": 0,
            "progressive_accuracy_pct": 0.0, "to_final_third_total": 0, "to_final_third_success": 0,
            "to_final_third_accuracy_pct": 0.0, "fwd": 0, "fwd_pct": 0.0,
            "bwd": 0, "bwd_pct": 0.0, "lat": 0, "lat_pct": 0.0,
            "pos_count": 0, "pos_pct": 0.0, "high_xt_pct": 0.0,
            "sum_dxt": 0.0, "total_p90": 0.0, "prog_p90": 0.0,
            "f3_p90": 0.0, "xt_p90": 0.0, "neg_xt_p90": 0.0,
            "minutes": mins, "long_acc_pct": 0.0, "high_xt_p90": 0.0, "dz_p90": 0.0,
        }
    successful = int(df["is_won"].sum())
    unsuccessful = total - successful
    accuracy = successful / total * 100.0
    progressive_total = int(df["progressive"].sum())
    progressive_unsuccessful = int((~df["is_won"] & df.apply(
        lambda r: is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]), axis=1)).sum())
    progressive_attempted = progressive_total + progressive_unsuccessful
    progressive_accuracy = (progressive_total / progressive_attempted * 100.0) if progressive_attempted else 0.0
    to_final_third = (df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)
    to_final_third_total = int(to_final_third.sum())
    to_final_third_success = int((to_final_third & df["is_won"]).sum())
    to_final_third_accuracy = (to_final_third_success / to_final_third_total * 100.0) if to_final_third_total else 0.0
    long_passes = df[df["pass_distance"] > 25.0]
    long_total = len(long_passes)
    long_success = int(long_passes["is_won"].sum())
    long_acc_pct = (long_success / long_total * 100.0) if long_total > 0 else 0.0
    dz_mask = df["is_won"] & ((df["x_end"] >= 100.0) | ((df["x_end"] >= 80.0) & (df["x_end"] < 100.0) & (df["y_end"] >= LANE_RIGHT_MAX) & (df["y_end"] < LANE_LEFT_MIN)))
    dz_passes = int(dz_mask.sum())
    fwd = int(df["is_forward"].sum())
    bwd = int(df["is_backward"].sum())
    lat = int(df["is_lateral"].sum())
    pos_count = int((df["is_won"] & (df["delta_xt_adj"] > 0)).sum())
    pos_pct = (pos_count / total * 100.0) if total > 0 else 0.0
    high_xt = int((df["delta_xt_adj"] > 0.1).sum())
    sum_dxt = float(df.loc[df["is_won"], "delta_xt_adj"].sum())
    neg_xt = float(df.loc[df["is_won"] & (df["delta_xt_adj"] < 0), "delta_xt_adj"].sum())
    return {
        "total_passes": total, "successful_passes": successful, "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 1), "progressive_attempted": progressive_attempted,
        "progressive_successful": progressive_total, "progressive_accuracy_pct": round(progressive_accuracy, 1),
        "to_final_third_total": to_final_third_total, "to_final_third_success": to_final_third_success,
        "to_final_third_accuracy_pct": round(to_final_third_accuracy, 1),
        "fwd": fwd, "fwd_pct": round(fwd / total * 100.0, 1), "bwd": bwd,
        "bwd_pct": round(bwd / total * 100.0, 1), "lat": lat, "lat_pct": round(lat / total * 100.0, 1),
        "pos_count": pos_count, "pos_pct": round(pos_pct, 1), "high_xt_pct": round(high_xt / total * 100.0, 1),
        "sum_dxt": round(sum_dxt, 3), "total_p90": round(total * p90_factor, 1),
        "prog_p90": round(progressive_total * p90_factor, 1), "f3_p90": round(to_final_third_success * p90_factor, 1),
        "xt_p90": round(sum_dxt * p90_factor, 3), "neg_xt_p90": round(neg_xt * p90_factor, 3),
        "minutes": mins, "long_acc_pct": round(long_acc_pct, 1), "high_xt_p90": round(high_xt * p90_factor, 1),
        "dz_p90": round(dz_passes * p90_factor, 1),
    }

def compute_match_scores(dfs_dict, defensive_dfs_dict=None):
    records = []
    for m_name, df_m in dfs_dict.items():
        s = compute_stats(df_m, m_name)
        if s['total_passes'] == 0:
            continue
        records.append({
            'match': m_name, 'xt_p90': s['xt_p90'], 'prog_p90': s['prog_p90'],
            'f3_p90': s['f3_p90'], 'pos_pct': s['pos_pct'], 'total_p90': s['total_p90'],
            'neg_xt_p90': s['neg_xt_p90'], 'accuracy_pct': s['accuracy_pct'],
            'long_acc_pct': s['long_acc_pct'], 'high_xt_p90': s['high_xt_p90'],
            'dz_p90': s['dz_p90'], 'prog_acc_pct': s['progressive_accuracy_pct'],
        })
    df_scores = pd.DataFrame(records)
    if df_scores.empty:
        return df_scores
    def normalize_fixed(series, val_min, val_max):
        clipped_series = series.clip(lower=val_min, upper=val_max)
        if val_max == val_min:
            return pd.Series([70.0] * len(series))
        return 40 + ((clipped_series - val_min) / (val_max - val_min)) * 60
    df_scores['xt_norm'] = normalize_fixed(df_scores['xt_p90'], val_min=0.05, val_max=0.45)
    df_scores['prog_norm'] = normalize_fixed(df_scores['prog_p90'], val_min=1.0, val_max=15.0)
    df_scores['f3_norm'] = normalize_fixed(df_scores['f3_p90'], val_min=1.0, val_max=15.0)
    df_scores['pos_pct_norm'] = normalize_fixed(df_scores['pos_pct'], val_min=25.0, val_max=75.0)
    df_scores['total_p90_norm'] = normalize_fixed(df_scores['total_p90'], val_min=10.0, val_max=85.0)
    df_scores['neg_xt_norm'] = normalize_fixed(df_scores['neg_xt_p90'], val_min=-0.15, val_max=0.00)
    df_scores['Grade'] = (df_scores['xt_norm']*0.30 + df_scores['prog_norm']*0.20 + df_scores['f3_norm']*0.20 +
                          df_scores['pos_pct_norm']*0.10 + df_scores['total_p90_norm']*0.10 + df_scores['neg_xt_norm']*0.10)
    if defensive_dfs_dict is not None:
        def_bonus_list = []; duels_p90_list = []; duels_won_pct_list = []
        interceptions_p90_list = []; duels_won_p90_list = []; int_xt_avg_list = []; funnel_p90_list = []
        for _, row in df_scores.iterrows():
            team = row['match'].split('(')[0].strip()
            bonus = 0; dp = 0; dwp = 0; ip = 0; dwp90 = 0; int_xt_avg_rounded = 0; funnel_p90_val = 0.0
            for def_name, def_df in defensive_dfs_dict.items():
                def_team = def_name.split('(')[0].strip()
                if def_team == team:
                    mins = get_match_minutes(def_name)
                    p90 = 90.0 / mins if mins > 0 else 1.0
                    duels_won = int(def_df["is_duel_won"].sum())
                    duels_lost = int(def_df["is_duel_lost"].sum())
                    total_duels = duels_won + duels_lost
                    dwp = (duels_won / total_duels * 100.0) if total_duels > 0 else 0
                    dp = round(total_duels * p90, 1)
                    dwp90 = round(duels_won * p90, 1)
                    interceptions = int(def_df["is_interception"].sum())
                    ip = round(interceptions * p90, 1)
                    funnel_count = int(def_df["in_funnel"].sum())
                    funnel_p90_val = round(funnel_count * p90, 1)
                    int_df = def_df[def_df["is_interception"]]
                    if len(int_df) > 0:
                        int_xt_vals = [xt_value(float(r["x"]), float(r["y"])) for _, r in int_df.iterrows()]
                        int_xt_avg = float(np.mean(int_xt_vals)) if int_xt_vals else 0
                    else:
                        int_xt_avg = 0
                    int_xt_avg_rounded = round(int_xt_avg, 3)
                    duel_bonus_val = (dwp / 100.0) * min(dp / 20.0, 1.0) * 5
                    int_bonus_val = min(ip / 10.0, 1.0) * (0.5 + int_xt_avg * 2) * 3
                    bonus = duel_bonus_val + int_bonus_val
                    break
            def_bonus_list.append(round(bonus, 1))
            duels_p90_list.append(dp); duels_won_pct_list.append(round(dwp, 1))
            interceptions_p90_list.append(ip); duels_won_p90_list.append(dwp90)
            int_xt_avg_list.append(int_xt_avg_rounded); funnel_p90_list.append(funnel_p90_val)
        df_scores['def_bonus'] = def_bonus_list; df_scores['duels_p90'] = duels_p90_list
        df_scores['duels_won_pct'] = duels_won_pct_list; df_scores['interceptions_p90'] = interceptions_p90_list
        df_scores['duels_won_p90'] = duels_won_p90_list; df_scores['int_xt_avg'] = int_xt_avg_list
        df_scores['funnel_p90'] = funnel_p90_list
    else:
        df_scores['def_bonus'] = 0.0; df_scores['duels_p90'] = 0.0; df_scores['duels_won_pct'] = 0.0
        df_scores['interceptions_p90'] = 0.0; df_scores['duels_won_p90'] = 0.0
        df_scores['int_xt_avg'] = 0.0; df_scores['funnel_p90'] = 0.0
    df_scores['pass_grade'] = df_scores['Grade'].round(1).copy()
    def _norm_def(s, lo, hi):
        clipped = s.clip(lower=lo, upper=hi)
        if hi <= lo: return pd.Series([70.0] * len(s))
        return 40 + ((clipped - lo) / (hi - lo)) * 60
    df_scores['duels_won_pct_norm'] = _norm_def(df_scores['duels_won_pct'], 0, 100)
    df_scores['int_xt_norm'] = _norm_def(df_scores['int_xt_avg'], 0, 0.30)
    df_scores['duels_won_p90_norm'] = _norm_def(df_scores['duels_won_p90'], 0, 15)
    df_scores['interceptions_p90_norm'] = _norm_def(df_scores['interceptions_p90'], 0, 15)
    df_scores['funnel_p90_norm'] = _norm_def(df_scores['funnel_p90'], 0, 15)
    df_scores['def_grade'] = (df_scores['duels_won_pct_norm']*0.30 + df_scores['funnel_p90_norm']*0.15 +
                              df_scores['int_xt_norm']*0.25 + df_scores['duels_won_p90_norm']*0.15 +
                              df_scores['interceptions_p90_norm']*0.15).round(1)
    df_scores['Grade'] = (df_scores['pass_grade']*0.75 + df_scores['def_grade']*0.25).round(1)
    return df_scores

def compute_defensive_stats(df: pd.DataFrame, match_name: str) -> dict:
    total_actions = len(df)
    if match_name == "All Matches":
        mins = sum(get_match_minutes(k) for k in defensive_dfs_by_match)
    else:
        mins = get_match_minutes(match_name)
    p90_factor = 90.0 / mins if mins > 0 else 1.0
    duels_won = int(df["is_duel_won"].sum())
    duels_lost = int(df["is_duel_lost"].sum())
    total_duels = duels_won + duels_lost
    duels_won_pct = (duels_won / total_duels * 100.0) if total_duels > 0 else 0.0
    interceptions = int(df["is_interception"].sum())
    attacking_half = df[df["is_attacking_half"]]
    actions_attacking = len(attacking_half)
    interceptions_attacking = int(attacking_half["is_interception"].sum())
    funnel_actions = int(df["in_funnel"].sum())
    return {
        "total_actions": total_actions, "total_actions_p90": round(total_actions * p90_factor, 1),
        "actions_attacking": actions_attacking, "actions_attacking_p90": round(actions_attacking * p90_factor, 1),
        "total_duels": total_duels, "duels_p90": round(total_duels * p90_factor, 1),
        "duels_won_pct": round(duels_won_pct, 1), "duels_won": duels_won,
        "interceptions": interceptions, "interceptions_p90": round(interceptions * p90_factor, 1),
        "interceptions_attacking": interceptions_attacking, "interceptions_attacking_p90": round(interceptions_attacking * p90_factor, 1),
        "funnel_actions": funnel_actions, "funnel_actions_p90": round(funnel_actions * p90_factor, 1),
    }

# UI HELPERS
def _safe_pct_diff(a: float, b: float) -> float:
    base = max(abs(b), 1.0)
    pct = (abs(a - b) / base) * 100.0
    return min(pct, 999.0)

def _arrow_html(val_game: float, val_avg: float) -> str:
    if np.isclose(val_game, val_avg, atol=1e-9): return ""
    if abs(val_game) < 1 and abs(val_avg) < 1: return ""
    if val_game > val_avg:
        pct = _safe_pct_diff(val_game, val_avg)
        return f'<span style="color:#34d399;font-size:9px"> +{pct:.0f}%</span>'
    else:
        pct = _safe_pct_diff(val_avg, val_game)
        return f'<span style="color:#f87171;font-size:9px"> -{pct:.0f}%</span>'

def section_card(title, border_color, items):
    bg = _hex_to_rgba(border_color, 0.55)
    bd = _hex_to_rgba(border_color, 0.30)
    html = f'<div style="background:linear-gradient(135deg,{bg},#1a1a2e);border:1px solid {bd};border-radius:10px;padding:16px 16px 8px 16px;margin-bottom:10px">'
    html += f'<div style="color:#e0e0f0;font-size:16px;font-weight:700;padding-bottom:8px;border-bottom:1px solid {bd};margin-bottom:10px">{title}</div>'
    html += f'<div style="display:flex;flex-direction:column;gap:6px">'
    for idx, item in enumerate(items):
        label = item[0]
        value = item[1]
        sub = item[2] if len(item) > 2 else ""
        tooltip = item[3] if len(item) > 3 else ""
        is_last = idx == len(items) - 1
        sep = "" if is_last else 'style="border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:10px;margin-bottom:10px"'
        html += f'<div {sep}>'
        html += f'<div style="text-align:center;display:flex;flex-direction:column;align-items:center;gap:2px">'
        if tooltip:
            label_html = f'{label}'
            label_html += f'<span style="cursor:help;margin-left:4px;font-size:9px;color:#8888aa;border:1px solid #8888aa;border-radius:50%;padding:0 5px">?</span>'
            html += f'<span style="color:#ffffff;font-size:15px" title="{tooltip}">{label_html}</span>'
        else:
            html += f'<span style="color:#ffffff;font-size:15px">{label}</span>'
        html += f'<span style="color:#ffffff;font-size:24px;font-weight:800">{value}</span>'
        if sub:
            html += f'<span style="color:#ffffff;font-size:11px">{sub}</span>'
        html += '</div>'
        html += '</div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

def cmp_section_card(title, border_color, items):
    bg = _hex_to_rgba(border_color, 0.55)
    bd = _hex_to_rgba(border_color, 0.30)
    html = f'<div style="background:linear-gradient(135deg,{bg},#1a1a2e);border:1px solid {bd};border-radius:10px;padding:16px 16px 8px 16px;margin-bottom:10px">'
    html += f'<div style="color:#e0e0f0;font-size:16px;font-weight:700;padding-bottom:8px;border-bottom:1px solid {bd};margin-bottom:10px">{title}</div>'
    html += f'<div style="display:flex;flex-direction:column;gap:6px">'
    for idx, item in enumerate(items):
        label = item[0]
        val_game = item[1]
        val_avg = item[2]
        disp_game = item[3] if len(item) > 3 else str(val_game)
        disp_avg = item[4] if len(item) > 4 else str(val_avg)
        tooltip = item[5] if len(item) > 5 else ""
        arrow = _arrow_html(float(val_game), float(val_avg))
        is_last = idx == len(items) - 1
        sep = "" if is_last else 'style="border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:10px;margin-bottom:10px"'
        html += f'<div {sep}>'
        html += f'<div style="text-align:center;display:flex;flex-direction:column;align-items:center;gap:2px">'
        if tooltip:
            label_html = f'{label}'
            label_html += f'<span style="cursor:help;margin-left:4px;font-size:9px;color:#8888aa;border:1px solid #8888aa;border-radius:50%;padding:0 5px">?</span>'
            html += f'<span style="color:#ffffff;font-size:15px" title="{tooltip}">{label_html}</span>'
        else:
            html += f'<span style="color:#ffffff;font-size:15px">{label}</span>'
        html += f'<span style="color:#ffffff;font-size:24px;font-weight:800">{disp_game}{arrow}</span>'
        html += f'<span style="color:#ffffff;font-size:11px">AVG: {disp_avg}</span>'
        html += '</div>'
        html += '</div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

# DRAW HELPERS (PITCH)
def _base_pitch(bg="#1a1a2e"):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=bg, line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor(bg)
    fig.set_dpi(FIG_DPI)
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#ffffff", lw=1.2, alpha=0.40, linestyle="--")
    ax.axvline(x=HALF_LINE_X, color="#ffffff", lw=0.7, alpha=0.12, linestyle="--")
    return fig, ax, pitch

def _attack_arrow(fig, has_cbar=False):
    ox = -0.04 if has_cbar else 0.0
    fig.patches.append(FancyArrowPatch(
        (0.44 + ox, 0.045), (0.56 + ox, 0.045), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=11, linewidth=1.6, color="#aaaaaa"))
    fig.text(0.50 + ox, 0.012, "Attacking Direction", ha="center", va="bottom",
             transform=fig.transFigure, fontsize=7.5, color="#aaaaaa")

def _save_fig(fig):
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)

def draw_pass_map(df):
    fig, ax, pitch = _base_pitch()
    has_crosses = "is_cross" in df.columns and df["is_cross"].any()
    for _, row in df.iterrows():
        is_cross = bool(row.get("is_cross", False))
        if is_cross:
            if row["is_won"]:
                color, alpha = COLOR_CROSS_WON, 0.90
            else:
                color, alpha = COLOR_CROSS_LOST, 0.85
        else:
            is_lost = not row["is_won"]
            is_prog = bool(row["progressive"])
            if is_lost: color, alpha = COLOR_FAIL, 0.72
            elif is_prog: color, alpha = COLOR_PROGRESSIVE, 0.88
            else: color, alpha = COLOR_SUCCESS, ALPHA_SUCCESS
        pitch.arrows(row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                     color=color, width=1.3, headwidth=2.0, headlength=2.0,
                     ax=ax, zorder=3, alpha=alpha)
        pitch.scatter(row["x_start"], row["y_start"], s=32, marker="o",
                      color=color, edgecolors="white", linewidths=0.6,
                      ax=ax, zorder=6, alpha=alpha)
    leg_handles = [
        Line2D([0], [0], color=COLOR_SUCCESS, lw=2.0, label="Completed", alpha=0.65),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.0, label="Progressive", alpha=0.90),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.0, label="Incomplete", alpha=0.90),
    ]
    if has_crosses:
        leg_handles.append(Line2D([0], [0], color=COLOR_CROSS_WON, lw=2.0, label="Cross Won", alpha=0.90))
        leg_handles.append(Line2D([0], [0], color=COLOR_CROSS_LOST, lw=2.0, label="Cross Lost", alpha=0.85))
    leg = ax.legend(handles=leg_handles, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                    frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
                    fontsize=6.5, labelspacing=0.35, borderpad=0.4)
    for t in leg.get_texts(): t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_corridor_heatmap(df):
    df_s = df[df["is_won"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)
    corridors = {"left": (LANE_LEFT_MIN, FIELD_Y), "center": (LANE_RIGHT_MAX, LANE_LEFT_MIN), "right": (0.0, LANE_RIGHT_MAX)}
    counts = {}
    for cname, (y0, y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            arr[i] = int(((df_s["x_end"] >= x0_) & (df_s["x_end"] < x1_) & (df_s["y_end"] >= y0) & (df_s["y_end"] < y1)).sum())
        counts[cname] = arr
    all_vals = np.concatenate([counts[c] for c in counts])
    vmax = max(1, int(all_vals.max()))
    cmap = LinearSegmentedColormap.from_list("wr", ["#ffffff", "#ffecec", "#ffbfbf", "#ff8080", "#ff3b3b", "#ff0000"])
    norm = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax * 0.35)
    fig, ax, pitch = _base_pitch()
    for cname, (y0, y1) in corridors.items():
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            value = counts[cname][i]
            ax.add_patch(Rectangle((x0_, y0), x1_ - x0_, y1 - y0,
                                   facecolor=cmap(norm(value)), edgecolor=(1, 1, 1, 0.12),
                                   lw=0.5, alpha=0.95, zorder=2))
            ax.text((x0_ + x1_) / 2, (y0 + y1) / 2, str(value), ha="center", va="center",
                    color="#000000" if value <= threshold else "#ffffff",
                    fontsize=9, fontweight="700" if value >= vmax * 0.5 else "600", zorder=4)
    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def _draw_comet_arrow(ax, x0, y0, x1, y1, color):
    segs = 12
    ts = np.linspace(0.0, 1.0, segs + 1)
    for i in range(segs):
        t0, t1 = ts[i], ts[i + 1]
        xa = x0 + (x1 - x0) * t0; ya = y0 + (y1 - y0) * t0
        xb = x0 + (x1 - x0) * t1; yb = y0 + (y1 - y0) * t1
        alpha = 0.85 * (0.15 + 0.85 * t1)
        lw = 2.5 * (0.80 + 0.20 * t1)
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw, alpha=alpha, zorder=4, solid_capstyle="round")
    ax.scatter(x0, y0, s=20, marker="o", facecolors="none", edgecolors=color, linewidths=1.5, zorder=5, alpha=0.85)
    ax.scatter(x1, y1, s=32, marker="o", facecolors=color, edgecolors="white", linewidths=0.9, zorder=6, alpha=0.85)

def draw_top_xt_map(df, top_n=5):
    fig, ax, pitch = _base_pitch()
    eligible = df[(df["is_won"]) & (df["delta_xt_adj"] > 0)]
    if not eligible.empty:
        top_passes = eligible.sort_values("delta_xt_adj", ascending=False).head(top_n).copy().reset_index(drop=True)
        for _, row in top_passes.iterrows():
            val = float(row["delta_xt_adj"])
            color = CMAP_TOP10(NORM_TOP10(np.clip(val, 0.05, 0.40)))
            _draw_comet_arrow(ax, float(row["x_start"]), float(row["y_start"]),
                              float(row["x_end"]), float(row["y_end"]), color)
        sm = plt.cm.ScalarMappable(cmap=CMAP_TOP10, norm=NORM_TOP10)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.02, shrink=0.60)
        cbar.set_label("Pass Impact", color="#ffffff", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig

# DEFENSIVE PITCH DRAW HELPERS
COLOR_DUEL_WON = "#10b981"
COLOR_DUEL_LOST = "#E07070"
COLOR_INTERCEPTION = "#2F80ED"

def draw_defensive_map(df):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        if row["is_duel_won"]: color, marker, s, alpha = COLOR_DUEL_WON, "o", 90, 0.85
        elif row["is_duel_lost"]: color, marker, s, alpha = COLOR_DUEL_LOST, "X", 100, 0.85
        else: color, marker, s, alpha = COLOR_INTERCEPTION, "^", 80, 0.85
        pitch.scatter(row["x"], row["y"], s=s, marker=marker, color=color,
                      edgecolors="white", linewidths=0.8, ax=ax, zorder=6, alpha=alpha)
    leg = ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DUEL_WON, markersize=7, label="Duel Won", alpha=0.90),
            Line2D([0], [0], marker="X", color="w", markerfacecolor=COLOR_DUEL_LOST, markersize=8, label="Duel Lost", alpha=0.90),
            Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_INTERCEPTION, markersize=7, label="Interception", alpha=0.90),
        ], loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
        fontsize=6.5, labelspacing=0.35, borderpad=0.4)
    for t in leg.get_texts(): t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_funnel_protection_map(df):
    fig, ax, pitch = _base_pitch()
    funnel_rect = Rectangle((0, PENALTY_AREA_Y_MIN), FUNNEL_X_EXTEND, PENALTY_AREA_Y_MAX - PENALTY_AREA_Y_MIN,
                            facecolor="#ffd700", edgecolor="#ffd700", lw=1.5, linestyle="--", alpha=0.12, zorder=2)
    ax.add_patch(funnel_rect)
    for _, row in df.iterrows():
        x, y = float(row["x"]), float(row["y"])
        in_funnel = bool(row.get("in_funnel", is_in_funnel_zone(x, y)))
        if in_funnel: marker, s, color, edge = "*", 120, "#ffd700", "#b8860b"
        else: marker, s, color, edge = "o", 60, "#888888", "#555555"
        pitch.scatter(x, y, s=s, marker=marker, color=color, edgecolors=edge,
                      linewidths=0.5, ax=ax, zorder=6, alpha=0.85)
    leg = ax.legend(
        handles=[
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffd700", markersize=9, label="Funnel Action", alpha=0.95),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888", markersize=6, label="Other Action", alpha=0.50),
        ], loc="upper left", bbox_to_anchor=(0.01, 0.99),
        frameon=True, facecolor="#1a1a2e", edgecolor="#444466",
        fontsize=6.5, labelspacing=0.35, borderpad=0.4)
    for t in leg.get_texts(): t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_defensive_heatmap(df):
    corridors = {"Left": (LANE_LEFT_MIN, FIELD_Y), "Center": (LANE_RIGHT_MAX, LANE_LEFT_MIN), "Right": (0.0, LANE_RIGHT_MAX)}
    corridor_data = {}
    for cname, (y0, y1) in corridors.items():
        mask = (df["y"] >= y0) & (df["y"] < y1)
        corr_df = df[mask]
        total = len(corr_df)
        duels_total = int(corr_df["is_duel"].sum())
        duels_won = int(corr_df["is_duel_won"].sum())
        corridor_data[cname] = {"count": total, "duels_won": duels_won, "duels_total": duels_total}
    all_counts = [d["count"] for d in corridor_data.values()]
    vmax = max(1, max(all_counts))
    cmap_def = LinearSegmentedColormap.from_list("def_corr", ["#ffffff", "#dbeafe", "#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a5f"])
    norm = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax * 0.35)
    fig, ax, pitch = _base_pitch()
    for cname, (y0, y1) in corridors.items():
        d = corridor_data[cname]
        value = d["count"]
        ax.add_patch(Rectangle((0, y0), FIELD_X, y1 - y0, facecolor=cmap_def(norm(value)),
                               edgecolor=(1, 1, 1, 0.15), lw=0.5, alpha=0.95, zorder=2))
        duel_pct = (d["duels_won"] / d["duels_total"] * 100) if d["duels_total"] > 0 else None
        if duel_pct is not None:
            label = f"{cname}\nTotal: {value}\nWon: {d['duels_won']}/{d['duels_total']} ({duel_pct:.0f}%)"
        else:
            label = f"{cname}\nTotal: {value}"
        ax.text(FIELD_X / 2, (y0 + y1) / 2, label, ha="center", va="center",
                color="#000000" if value <= threshold else "#ffffff", fontsize=9, fontweight="600", zorder=4)
    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff", lw=0.5, alpha=0.20, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", lw=0.5, alpha=0.20, linestyle="--", zorder=3)
    _attack_arrow(fig)
    return _save_fig(fig), fig

# SIDEBAR
st.sidebar.markdown("""
<div style="text-align:center;padding:10px 0">
    <span style="font-size:24px;font-weight:800;color:#00d2ff;letter-spacing:1px">PASS STATS</span><br>
    <span style="font-size:13px;color:#8888bb">Dashboard</span><br>
    <span style="font-size:28px;font-weight:800;color:#ffffff;line-height:1.2">2026 Season</span><br>
    <span style="font-size:16px;color:#c0c0d0">Hudson Cicala</span>
</div>
""", unsafe_allow_html=True)

img_path = "Captura de tela 2026-06-02 154425.png"
if os.path.exists(img_path):
    st.sidebar.image(img_path, use_container_width=True)

st.sidebar.markdown("""
<div style="text-align:center;padding:10px 0;border-top:1px solid rgba(255,255,255,0.05);margin-top:6px">
    <span style="font-size:11px;color:#6666aa">Fullback · Center-Back</span><br>
    <span style="font-size:11px;color:#6666aa">Passing & Defensive Analysis</span>
</div>
""", unsafe_allow_html=True)

num_matches = len(dfs_by_match)
all_match_stats = [compute_stats(dfs_by_match[m], m) for m in dfs_by_match]

# MAIN SECTION
st.markdown("### Overall Performance Summary")

if num_matches > 0:
    total_passes_all = sum(s['total_passes'] for s in all_match_stats)
    total_succ_all = sum(s['successful_passes'] for s in all_match_stats)
    total_prog_all = sum(s['progressive_successful'] for s in all_match_stats)
    total_f3_all = sum(s['to_final_third_success'] for s in all_match_stats)
    total_pos_all = sum(s['pos_count'] for s in all_match_stats)
    total_xt_all = sum(s['sum_dxt'] for s in all_match_stats)
    avg_acc = sum(s['accuracy_pct'] for s in all_match_stats) / num_matches
    avg_prog_p90 = sum(s['prog_p90'] for s in all_match_stats) / num_matches
    avg_f3_p90 = sum(s['f3_p90'] for s in all_match_stats) / num_matches
    avg_pos_pct = sum(s['pos_pct'] for s in all_match_stats) / num_matches
    avg_xt_p90 = sum(s['xt_p90'] for s in all_match_stats) / num_matches
    avg_total_p90 = sum(s['total_p90'] for s in all_match_stats) / num_matches

    st.markdown("### Passes")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        section_card("Overview", C_BLUE_PASTEL, [
            ("Passes p90", f"{avg_total_p90:.1f}", f"Total: {total_passes_all}"),
            ("Successful %", f"{avg_acc:.1f}%", f"Total: {total_succ_all}"),
        ])
    with col_s2:
        section_card("Advanced", C_GREEN_PASTEL, [
            ("Progressive p90", f"{avg_prog_p90:.1f}", f"Total: {total_prog_all}"),
            ("Final Third p90", f"{avg_f3_p90:.1f}", f"Total: {total_f3_all}"),
        ])
    with col_s3:
        section_card("Impact", C_AMBER_PASTEL, [
            ("% Positive Impact", f"{avg_pos_pct:.1f}%", f"Total: {total_pos_all}",
             "Passes that generated a positive impact based on where they ended on the field"),
            ("Pass Impact Value", f"{avg_xt_p90:.1f}", f"Total: {total_xt_all:.1f}",
             "Calculation used to evaluate the offensive value added by a pass."),
        ])

    st.markdown("<br>", unsafe_allow_html=True)

    defensive_num_matches = len(defensive_dfs_by_match)
    defensive_all_stats = [compute_defensive_stats(defensive_dfs_by_match[m], m) for m in defensive_dfs_by_match]

    if defensive_num_matches > 0:
        total_def_actions_all = sum(s['total_actions'] for s in defensive_all_stats)
        total_def_att_all = sum(s['actions_attacking'] for s in defensive_all_stats)
        total_duels_all = sum(s['total_duels'] for s in defensive_all_stats)
        total_duels_won_all = sum(s['duels_won'] for s in defensive_all_stats)
        total_interceptions_all = sum(s['interceptions'] for s in defensive_all_stats)
        total_int_att_all = sum(s['interceptions_attacking'] for s in defensive_all_stats)
        avg_def_actions_p90 = sum(s['total_actions_p90'] for s in defensive_all_stats) / defensive_num_matches
        avg_def_att_p90 = sum(s['actions_attacking_p90'] for s in defensive_all_stats) / defensive_num_matches
        avg_duels_p90 = sum(s['duels_p90'] for s in defensive_all_stats) / defensive_num_matches
        avg_duels_won_pct = sum(s['duels_won_pct'] for s in defensive_all_stats) / defensive_num_matches
        avg_interceptions_p90 = sum(s['interceptions_p90'] for s in defensive_all_stats) / defensive_num_matches
        avg_int_att_p90 = sum(s['interceptions_attacking_p90'] for s in defensive_all_stats) / defensive_num_matches

        st.markdown("### Defensive Actions")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            section_card("General", C_BLUE_PASTEL, [
                ("Defensive Actions p90", f"{avg_def_actions_p90:.1f}", f"Total: {total_def_actions_all}"),
                ("Actions in Opp. Field p90", f"{avg_def_att_p90:.1f}", f"Total: {total_def_att_all}"),
            ])
        with col_d2:
            section_card("Duels", C_GREEN_PASTEL, [
                ("Defensive Duels p90", f"{avg_duels_p90:.1f}", f"Total: {total_duels_all}"),
                ("% Duels Won", f"{avg_duels_won_pct:.1f}%", f"({total_duels_won_all}/{total_duels_all})"),
            ])
        with col_d3:
            section_card("Interceptions", C_AMBER_PASTEL, [
                ("Interceptions p90", f"{avg_interceptions_p90:.1f}", f"Total: {total_interceptions_all}"),
                ("Interceptions in Opp Field p90", f"{avg_int_att_p90:.1f}", f"Total: {total_int_att_all}"),
            ])

    st.markdown(f'<div style="text-align:right;color:#555588;font-size:10px;padding-right:4px">{num_matches} matches collected</div>', unsafe_allow_html=True)

# --- PASS MAPS AND DETAILS SECTION ---
st.markdown("---")
st.markdown("### Match Details - Passes")

col_f1, col_f2 = st.columns(2)
with col_f1:
    pass_match_options = ["All Matches"] + ACTIVE_PASS_MATCHES
    selected_match = st.selectbox("Select Match", options=pass_match_options, index=0, key="pass_match")
with col_f2:
    pass_filter = st.radio(
        "Pass Type",
        ["All", "Successful", "Unsuccessful", "Progressive", "Final Third", "Crosses"],
        index=0, horizontal=True, key="pass_filter"
    )

if selected_match == "All Matches":
    df_game_filtered = pd.concat(dfs_by_match.values(), ignore_index=True)
    match_name_for_stats = "All Matches"
else:
    df_game_filtered = dfs_by_match[selected_match].copy()
    match_name_for_stats = selected_match

def apply_filter(df):
    if pass_filter == "Successful": return df[df["is_won"]].copy()
    if pass_filter == "Unsuccessful": return df[~df["is_won"]].copy()
    if pass_filter == "Progressive": return df[df["progressive"]].copy()
    if pass_filter == "Final Third": return df[(df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)].copy()
    if pass_filter == "Crosses": return df[df["is_cross"]].copy()
    return df.copy()

df_game = apply_filter(df_game_filtered)
s_game = compute_stats(df_game, match_name_for_stats)

s_avg = {}
if num_matches > 0:
    for k in all_match_stats[0].keys():
        if isinstance(all_match_stats[0][k], (int, float)):
            s_avg[k] = sum(s[k] for s in all_match_stats) / num_matches
        else:
            s_avg[k] = 0
else:
    s_avg = s_game.copy()

force_avg = selected_match == "All Matches"
if force_avg:
    s_game = s_avg.copy()

img_pm_game, fig_pm_game = draw_pass_map(df_game); plt.close(fig_pm_game)
img_ht_game, fig_ht_game = draw_corridor_heatmap(df_game); plt.close(fig_ht_game)
img_xt_game, fig_xt_game = draw_top_xt_map(df_game, top_n=5); plt.close(fig_xt_game)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown('<div style="color:#a0a0b5;font-size:11px;font-weight:600;padding-bottom:4px">Pass Map</div>', unsafe_allow_html=True)
    st.image(img_pm_game, use_container_width=True)
with col_m2:
    st.markdown('<div style="color:#a0a0b5;font-size:11px;font-weight:600;padding-bottom:4px">Zone Heatmap</div>', unsafe_allow_html=True)
    st.image(img_ht_game, use_container_width=True)
with col_m3:
    st.markdown('<div style="color:#a0a0b5;font-size:11px;font-weight:600;padding-bottom:4px">Top 5 Pass Impact</div>', unsafe_allow_html=True)
    st.image(img_xt_game, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

col_s1, col_s2, col_s3 = st.columns(3)
if force_avg:
    with col_s1:
        section_card("Pass Overview", C_BLUE_PASTEL, [
            ("Total Passes", f"{s_game['total_p90']:.1f}"),
            ("Successful %", f"{s_game['accuracy_pct']:.1f}%"),
        ])
    with col_s2:
        section_card("Advanced", C_GREEN_PASTEL, [
            ("Progressive", f"{s_game['prog_p90']:.1f}"),
            ("Final Third", f"{s_game['f3_p90']:.1f}"),
        ])
    with col_s3:
        section_card("Impact", C_AMBER_PASTEL, [
            ("% Positive Impact", f"{s_game['pos_pct']:.1f}%"),
            ("Pass Impact Value", f"{s_game['xt_p90']:.1f}"),
        ])
else:
    with col_s1:
        cmp_section_card("Pass Overview", C_BLUE_PASTEL, [
            ("Total Passes", s_game["total_p90"], f"{s_avg['total_p90']:.1f}"),
            ("Successful %", s_game["accuracy_pct"], s_avg["accuracy_pct"],
             f"{s_game['accuracy_pct']:.1f}%", f"{s_avg['accuracy_pct']:.1f}%"),
        ])
    with col_s2:
        cmp_section_card("Advanced", C_GREEN_PASTEL, [
            ("Progressive", s_game["prog_p90"], f"{s_avg['prog_p90']:.1f}"),
            ("Final Third", s_game["f3_p90"], f"{s_avg['f3_p90']:.1f}"),
        ])
    with col_s3:
        cmp_section_card("Impact", C_AMBER_PASTEL, [
            ("% Positive Impact", s_game["pos_pct"], s_avg["pos_pct"],
             f"{s_game['pos_pct']:.1f}%", f"{s_avg['pos_pct']:.1f}%",
             "Passes that generated a positive impact based on where they ended on the field"),
            ("Pass Impact Value", s_game["xt_p90"], s_avg["xt_p90"],
             f"{s_game['xt_p90']:.1f}", f"{s_avg['xt_p90']:.1f}",
             "Calculation used to define the value of pass impact based on expected threat (xT) progression"),
        ])

# --- DEFENSIVE MAPS AND DETAILS SECTION ---
st.markdown("---")
st.markdown("### Match Details - Defensive Actions")

col_df1, col_df2 = st.columns(2)
with col_df1:
    def_match_options = ["All Matches"] + ACTIVE_DEF_MATCHES
    selected_def_match = st.selectbox("Select Match", options=def_match_options, index=0, key="def_match")
with col_df2:
    def_type_filter = st.radio("Filter Type", ["All", "Duels Only", "Interceptions Only"], horizontal=True, key="def_type_filter")

if selected_def_match == "All Matches":
    df_def_game_raw = pd.concat(defensive_dfs_by_match.values(), ignore_index=True)
    def_match_name_for_stats = "All Matches"
else:
    df_def_game_raw = defensive_dfs_by_match[selected_def_match].copy()
    def_match_name_for_stats = selected_def_match

if def_type_filter == "Duels Only":
    df_def_game = df_def_game_raw[df_def_game_raw["is_duel"]].copy()
elif def_type_filter == "Interceptions Only":
    df_def_game = df_def_game_raw[df_def_game_raw["is_interception"]].copy()
else:
    df_def_game = df_def_game_raw.copy()

d_game = compute_defensive_stats(df_def_game, def_match_name_for_stats)
def_all = [compute_defensive_stats(defensive_dfs_by_match[m], m) for m in defensive_dfs_by_match]
d_avg = {}
if len(def_all) > 0:
    for k in def_all[0].keys():
        if isinstance(def_all[0][k], (int, float)):
            d_avg[k] = sum(s[k] for s in def_all) / len(def_all)
        else:
            d_avg[k] = 0
else:
    d_avg = d_game.copy()

force_avg_def = selected_def_match == "All Matches"
if force_avg_def:
    d_game = d_avg.copy()

img_def_map, fig_def_map = draw_defensive_map(df_def_game); plt.close(fig_def_map)
img_def_hm, fig_def_hm = draw_defensive_heatmap(df_def_game); plt.close(fig_def_hm)
img_funnel, fig_funnel = draw_funnel_protection_map(df_def_game); plt.close(fig_funnel)

col_dm1, col_dm2, col_dm3 = st.columns(3)
with col_dm1:
    st.markdown('<div style="color:#a0a0b5;font-size:11px;font-weight:600;padding-bottom:4px">Defensive Actions Map</div>', unsafe_allow_html=True)
    st.image(img_def_map, use_container_width=True)
with col_dm2:
    st.markdown('<div style="color:#a0a0b5;font-size:11px;font-weight:600;padding-bottom:4px">Defensive Heatmap</div>', unsafe_allow_html=True)
    st.image(img_def_hm, use_container_width=True)
with col_dm3:
    st.markdown('<div style="color:#a0a0b5;font-size:11px;font-weight:600;padding-bottom:4px">Funnel Protection Actions</div>', unsafe_allow_html=True)
    st.image(img_funnel, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

col_ds1, col_ds2, col_ds3 = st.columns(3)
if force_avg_def:
    with col_ds1:
        section_card("General", C_BLUE_PASTEL, [
            ("Defensive Actions", f"{d_game['total_actions_p90']:.1f}"),
            ("Actions in Opp. Field", f"{d_game['actions_attacking_p90']:.1f}"),
        ])
    with col_ds2:
        section_card("Duels", C_GREEN_PASTEL, [
            ("Defensive Duels", f"{d_game['duels_p90']:.1f}"),
            ("% Duels Won", f"{d_game['duels_won_pct']:.1f}%"),
        ])
    with col_ds3:
        section_card("Interceptions", C_AMBER_PASTEL, [
            ("Interceptions", f"{d_game['interceptions_p90']:.1f}"),
            ("Interceptions in Opp Field", f"{d_game['interceptions_attacking_p90']:.1f}"),
        ])
else:
    with col_ds1:
        cmp_section_card("General", C_BLUE_PASTEL, [
            ("Defensive Actions", d_game["total_actions_p90"], f"{d_avg['total_actions_p90']:.1f}"),
            ("Actions in Opp. Field", d_game["actions_attacking_p90"], f"{d_avg['actions_attacking_p90']:.1f}"),
        ])
    with col_ds2:
        cmp_section_card("Duels", C_GREEN_PASTEL, [
            ("Defensive Duels", d_game["duels_p90"], f"{d_avg['duels_p90']:.1f}"),
            ("% Duels Won", d_game["duels_won_pct"], d_avg["duels_won_pct"],
             f"{d_game['duels_won_pct']:.1f}%", f"{d_avg['duels_won_pct']:.1f}%"),
        ])
    with col_ds3:
        cmp_section_card("Interceptions", C_AMBER_PASTEL, [
            ("Interceptions", d_game["interceptions_p90"], f"{d_avg['interceptions_p90']:.1f}"),
            ("Interceptions in Opp Field", d_game["interceptions_attacking_p90"], f"{d_avg['interceptions_attacking_p90']:.1f}"),
        ])
