import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap
import plotly.graph_objects as go
import re
import os
import math
from pathlib import Path

st.set_page_config(layout="wide", page_title="Stats — Dashboard")

# ============================================================
# CONSTANTS
# ============================================================
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
C_BLUE_PASTEL = "#7fb3d8"
C_GREEN_PASTEL = "#8abb70"
C_AMBER_PASTEL = "#dbbc6a"
CMAP_TOP10 = LinearSegmentedColormap.from_list("top10", ["#fef08a", "#f97316", "#b91c1c"])
NORM_TOP10 = Normalize(vmin=0.05, vmax=0.40)
NX_XT, NY_XT = 16, 12
D_REF, D_SCALE, BONUS_CAP = 10.0, 20.0, 0.60
LATERAL_MIN_DIST = 12.0
PENALTY_AREA_X = 18.0
FUNNEL_X_EXTEND = 33.0
PENALTY_AREA_Y_MIN = 18.0
PENALTY_AREA_Y_MAX = 62.0

# ============================================================
# HELPERS
# ============================================================
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

# ============================================================
# DATA — PASSES
# ============================================================
BASE_MATCHES_DATA = {
    "Real Salt Lake": [
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
    "Vancouver Whitecaps": [
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
    "LAFC": [
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

# ============================================================
# DATA — DEFENSIVE ACTIONS
# ============================================================
DEFENSIVE_MATCHES_DATA = {
    "Real Salt Lake": [
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
    "Vancouver Whitecaps": [
        ("DUEL_WON", 49.19, 64.77),
        ("INTERCEPTION", 75.96, 63.61),
        ("INTERCEPTION", 27.25, 50.97),
        ("INTERCEPTION", 35.23, 66.76),
    ],
    "LAFC": [
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

# ============================================================
# BUILD DATAFRAMES
# ============================================================
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

# ============================================================
# STATS COMPUTATION
# ============================================================
def compute_stats(df: pd.DataFrame, match_name: str) -> dict:
    total = len(df)
    mins = get_match_minutes(match_name)
    p90_factor = 90.0 / mins if mins > 0 else 1.0
    if total == 0:
        return {
            "total_passes": 0, "successful_passes": 0, "unsuccessful_passes": 0,
            "accuracy_pct": 0.0, "progressive_attempted": 0, "progressive_successful": 0,
            "progressive_accuracy_pct": 0.0, "to_final_third_total": 0, "to_final_third_success": 0,
            "to_final_third_accuracy_pct": 0.0, "fwd": 0, "fwd_pct": 0.0, "bwd": 0, "bwd_pct": 0.0,
            "lat": 0, "lat_pct": 0.0, "pos_count": 0, "pos_pct": 0.0, "high_xt_pct": 0.0,
            "sum_dxt": 0.0, "total_p90": 0.0, "prog_p90": 0.0, "f3_p90": 0.0, "xt_p90": 0.0,
            "neg_xt_p90": 0.0, "minutes": mins, "long_acc_pct": 0.0, "high_xt_p90": 0.0, "dz_p90": 0.0,
        }
    successful = int(df["is_won"].sum())
    unsuccessful = total - successful
    accuracy = successful / total * 100.0
    progressive_total = int(df["progressive"].sum())
    progressive_unsuccessful = int(
        (~df["is_won"] & df.apply(
            lambda r: is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
            axis=1
        )).sum()
    )
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
    dz_mask = df["is_won"] & (
        (df["x_end"] >= 100.0) |
        ((df["x_end"] >= 80.0) & (df["x_end"] < 100.0) & (df["y_end"] >= LANE_RIGHT_MAX) & (df["y_end"] < LANE_LEFT_MIN))
    )
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
        "fwd": fwd, "fwd_pct": round(fwd / total * 100.0, 1),
        "bwd": bwd, "bwd_pct": round(bwd / total * 100.0, 1),
        "lat": lat, "lat_pct": round(lat / total * 100.0, 1),
        "pos_count": pos_count, "pos_pct": round(pos_pct, 1),
        "high_xt_pct": round(high_xt / total * 100.0, 1),
        "sum_dxt": round(sum_dxt, 3), "total_p90": round(total * p90_factor, 1),
        "prog_p90": round(progressive_total * p90_factor, 1),
        "f3_p90": round(to_final_third_success * p90_factor, 1),
        "xt_p90": round(sum_dxt * p90_factor, 3),
        "neg_xt_p90": round(neg_xt * p90_factor, 3),
        "minutes": mins, "long_acc_pct": round(long_acc_pct, 1),
        "high_xt_p90": round(high_xt * p90_factor, 1),
        "dz_p90": round(dz_passes * p90_factor, 1),
    }

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
        "interceptions_attacking": interceptions_attacking,
        "interceptions_attacking_p90": round(interceptions_attacking * p90_factor, 1),
        "funnel_actions": funnel_actions, "funnel_actions_p90": round(funnel_actions * p90_factor, 1),
    }

# ============================================================
# UI — SECTION CARDS
# ============================================================
def section_card(title, border_color, items):
    bg = _hex_to_rgba(border_color, 0.55)
    bd = _hex_to_rgba(border_color, 0.30)
    html = f'<div style="background:linear-gradient(135deg,{bg},rgba(0,0,0,0.08));border-radius:12px;padding:14px 16px;border:1px solid {bd};margin-bottom:10px;box-shadow:0 2px 8px rgba(0,0,0,0.15);">'
    html += f'<div style="font-size:13px;font-weight:700;color:rgba(255,255,255,0.90);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:6px;">{title}</div>'
    html += '<div>'
    for idx, item in enumerate(items):
        label = item[0]
        value = item[1]
        sub = item[2] if len(item) > 2 else ""
        tooltip = item[3] if len(item) > 3 else ""
        is_last = idx == len(items) - 1
        sep = "" if is_last else 'style="border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:10px;margin-bottom:10px"'
        html += f'<div {sep}>'
        html += '<div style="display:flex;justify-content:space-between;align-items:center;">'
        if tooltip:
            html += f'<span style="font-size:13px;color:rgba(255,255,255,0.60);">{label} <span style="display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border-radius:50%;background:rgba(255,255,255,0.10);font-size:9px;cursor:default;color:rgba(255,255,255,0.45);margin-left:2px;" title="{tooltip}">?</span></span>'
        else:
            html += f'<span style="font-size:13px;color:rgba(255,255,255,0.60);">{label}</span>'
        html += f'<span style="font-size:18px;font-weight:700;color:white;">{value}</span>'
        html += '</div>'
        if sub:
            html += f'<div style="font-size:11px;color:rgba(255,255,255,0.40);margin-top:2px;text-align:right;">{sub}</div>'
        html += '</div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

def _safe_pct_diff(a: float, b: float) -> float:
    base = max(abs(b), 1.0)
    pct = (abs(a - b) / base) * 100.0
    return min(pct, 999.0)

def _arrow_html(val_game: float, val_avg: float) -> str:
    if np.isclose(val_game, val_avg, atol=1e-9):
        return ""
    if abs(val_game) < 1 and abs(val_avg) < 1:
        return ""
    if val_game > val_avg:
        pct = _safe_pct_diff(val_game, val_avg)
        return f' ▲ +{pct:.0f}%'
    else:
        pct = _safe_pct_diff(val_avg, val_game)
        return f' ▼ -{pct:.0f}%'

def cmp_section_card(title, border_color, items):
    bg = _hex_to_rgba(border_color, 0.55)
    bd = _hex_to_rgba(border_color, 0.30)
    html = f'<div style="background:linear-gradient(135deg,{bg},rgba(0,0,0,0.08));border-radius:12px;padding:14px 16px;border:1px solid {bd};margin-bottom:10px;box-shadow:0 2px 8px rgba(0,0,0,0.15);">'
    html += f'<div style="font-size:13px;font-weight:700;color:rgba(255,255,255,0.90);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:6px;">{title}</div>'
    html += '<div>'
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
        html += '<div style="display:flex;justify-content:space-between;align-items:center;">'
        if tooltip:
            html += f'<span style="font-size:13px;color:rgba(255,255,255,0.60);">{label} <span style="display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border-radius:50%;background:rgba(255,255,255,0.10);font-size:9px;cursor:default;color:rgba(255,255,255,0.45);margin-left:2px;" title="{tooltip}">?</span></span>'
        else:
            html += f'<span style="font-size:13px;color:rgba(255,255,255,0.60);">{label}</span>'
        html += f'<span style="font-size:18px;font-weight:700;color:white;">{disp_game}{arrow}</span>'
        html += '</div>'
        html += f'<div style="font-size:11px;color:rgba(255,255,255,0.40);margin-top:2px;text-align:right;">AVG: {disp_avg}</div>'
        html += '</div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# PITCH DRAWING FUNCTIONS
# ============================================================
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
        (0.44 + ox, 0.045), (0.56 + ox, 0.045),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=11,
        linewidth=1.6, color="#aaaaaa"
    ))
    fig.text(0.50 + ox, 0.012, "Attacking Direction", ha="center", va="bottom",
             transform=fig.transFigure, fontsize=7.5, color="#aaaaaa")

def _save_fig(fig):
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def draw_pass_map(df):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        is_lost = not row["is_won"]
        is_prog = bool(row["progressive"])
        if is_lost:
            color, alpha = COLOR_FAIL, 0.72
        elif is_prog:
            color, alpha = COLOR_PROGRESSIVE, 0.88
        else:
            color, alpha = COLOR_SUCCESS, ALPHA_SUCCESS
        pitch.arrows(row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                     color=color, width=1.3, headwidth=2.0, headlength=2.0, ax=ax, zorder=3, alpha=alpha)
        pitch.scatter(row["x_start"], row["y_start"], s=32, marker="o",
                      color=color, edgecolors="white", linewidths=0.6, ax=ax, zorder=6, alpha=alpha)
    leg_handles = [
        Line2D([0], [0], color=COLOR_SUCCESS, lw=2.0, label="Completed", alpha=0.65),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.0, label="Progressive", alpha=0.90),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.0, label="Incomplete", alpha=0.90),
    ]
    leg = ax.legend(handles=leg_handles, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                    frameon=True, facecolor="#1a1a2e", edgecolor="#444466", fontsize=6.5,
                    labelspacing=0.35, borderpad=0.4)
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig)

def draw_corridor_heatmap(df):
    df_s = df[df["is_won"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)
    corridors = {
        "left": (LANE_LEFT_MIN, FIELD_Y),
        "center": (LANE_RIGHT_MAX, LANE_LEFT_MIN),
        "right": (0.0, LANE_RIGHT_MAX)
    }
    counts = {}
    for cname, (y0, y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            arr[i] = int(((df_s["x_end"] >= x0_) & (df_s["x_end"] < x1_) &
                           (df_s["y_end"] >= y0) & (df_s["y_end"] < y1)).sum())
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
    return _save_fig(fig)

def _draw_comet_arrow(ax, x0, y0, x1, y1, color):
    segs = 12
    ts = np.linspace(0.0, 1.0, segs + 1)
    for i in range(segs):
        t0, t1 = ts[i], ts[i + 1]
        xa = x0 + (x1 - x0) * t0
        ya = y0 + (y1 - y0) * t0
        xb = x0 + (x1 - x0) * t1
        yb = y0 + (y1 - y0) * t1
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
    return _save_fig(fig)

def draw_cross_map(df):
    fig, ax, pitch = _base_pitch()
    crosses = df[df["is_cross"]].copy()
    if len(crosses) == 0:
        ax.text(FIELD_X/2, FIELD_Y/2, "No crosses in this selection", ha="center", va="center",
                color="#888888", fontsize=12, fontstyle="italic")
        _attack_arrow(fig)
        return _save_fig(fig)
    for _, row in crosses.iterrows():
        if row["is_won"]:
            color, alpha = COLOR_CROSS_WON, 0.85
        else:
            color, alpha = COLOR_CROSS_LOST, 0.85
        pitch.arrows(row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                     color=color, width=1.6, headwidth=2.5, headlength=2.5, ax=ax, zorder=4, alpha=alpha)
        pitch.scatter(row["x_start"], row["y_start"], s=40, marker="d",
                      color=color, edgecolors="white", linewidths=0.8, ax=ax, zorder=6, alpha=alpha)
    leg_handles = [
        Line2D([0], [0], color=COLOR_CROSS_WON, lw=2.0, label="Cross Completed", alpha=0.85),
        Line2D([0], [0], color=COLOR_CROSS_LOST, lw=2.0, label="Cross Incomplete", alpha=0.85),
    ]
    leg = ax.legend(handles=leg_handles, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                    frameon=True, facecolor="#1a1a2e", edgecolor="#444466", fontsize=6.5,
                    labelspacing=0.35, borderpad=0.4)
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig)

def draw_defensive_map(df):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        if row["is_duel_won"]:
            color, marker, s, alpha = COLOR_CROSS_WON, "o", 90, 0.85
        elif row["is_duel_lost"]:
            color, marker, s, alpha = COLOR_FAIL, "X", 100, 0.85
        else:
            color, marker, s, alpha = COLOR_PROGRESSIVE, "^", 80, 0.85
        pitch.scatter(row["x"], row["y"], s=s, marker=marker, color=color,
                      edgecolors="white", linewidths=0.8, ax=ax, zorder=6, alpha=alpha)
    leg = ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CROSS_WON, markersize=7, label="Duel Won", alpha=0.90),
            Line2D([0], [0], marker="X", color="w", markerfacecolor=COLOR_FAIL, markersize=8, label="Duel Lost", alpha=0.90),
            Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_PROGRESSIVE, markersize=7, label="Interception", alpha=0.90),
        ],
        loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True,
        facecolor="#1a1a2e", edgecolor="#444466", fontsize=6.5,
        labelspacing=0.35, borderpad=0.4
    )
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<h1 style="color:#ffffff;font-size:24px;font-weight:700;text-align:center;margin-bottom:4px;">📊 Stats Dashboard</h1>',
        unsafe_allow_html=True
    )
    img_path = "PHOTO-2025-10-24-21-10-55-2-e1761676493155.jpg"
    if os.path.exists(img_path):
        st.sidebar.image(img_path, use_container_width=True)
    st.sidebar.markdown(
        '<h2 style="color:#ffffff;font-size:18px;font-weight:600;text-align:center;margin-top:8px;">Azyk Gomez-Carmona</h2>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<p style="color:rgba(255,255,255,0.60);font-size:14px;text-align:center;">Right-Back</p>',
        unsafe_allow_html=True
    )

# ============================================================
# MAIN — OVERALL PERFORMANCE SUMMARY
# ============================================================
num_matches = len(dfs_by_match)
all_match_stats = [compute_stats(dfs_by_match[m], m) for m in dfs_by_match]
defensive_num_matches = len(defensive_dfs_by_match)
defensive_all_stats = [compute_defensive_stats(defensive_dfs_by_match[m], m) for m in defensive_dfs_by_match]

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
            ("Pass Impact Value", f"{avg_xt_p90:.3f}", f"Total: {total_xt_all:.3f}",
             "Calculation used to evaluate the offensive value added by a pass."),
        ])

    st.markdown("<br>", unsafe_allow_html=True)

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
            ("Interceptions in Opp. Field p90", f"{avg_int_att_p90:.1f}", f"Total: {total_int_att_all}"),
        ])

    st.markdown(f"<p style='color:rgba(255,255,255,0.40);font-size:12px;text-align:right;'>{num_matches} matches collected</p>", unsafe_allow_html=True)

# ============================================================
# MATCH DETAILS — PASSES
# ============================================================
st.markdown("---")
st.markdown("### Match Details — Passes")

col_f1, col_f2 = st.columns(2)
with col_f1:
    pass_match_options = ["All Matches"] + ACTIVE_PASS_MATCHES
    selected_match = st.selectbox("Select Match", options=pass_match_options, index=0, key="pass_match")
with col_f2:
    pass_filter = st.radio(
        "Pass Type",
        ["All", "Successful", "Unsuccessful", "Progressive", "Final Third"],
        index=0, horizontal=True, key="pass_filter"
    )

if selected_match == "All Matches":
    df_game_filtered = pd.concat(dfs_by_match.values(), ignore_index=True)
    match_name_for_stats = "All Matches"
else:
    df_game_filtered = dfs_by_match[selected_match].copy()
    match_name_for_stats = selected_match

def apply_filter(df):
    has_cross_col = "is_cross" in df.columns
    df_no_cross = df[~df["is_cross"]].copy() if has_cross_col else df.copy()
    if pass_filter == "All":
        return df_no_cross
    elif pass_filter == "Successful":
        return df_no_cross[df_no_cross["is_won"]]
    elif pass_filter == "Unsuccessful":
        return df_no_cross[~df_no_cross["is_won"]]
    elif pass_filter == "Progressive":
        return df_no_cross[df_no_cross["progressive"]]
    elif pass_filter == "Final Third":
        return df_no_cross[(df_no_cross["x_start"] < FINAL_THIRD_LINE_X) & (df_no_cross["x_end"] >= FINAL_THIRD_LINE_X)]
    return df_no_cross

df_game = apply_filter(df_game_filtered)
s_game = compute_stats(df_game, match_name_for_stats)

# Compute averages
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

# Generate maps
img_pm_game = draw_pass_map(df_game)
img_ht_game = draw_corridor_heatmap(df_game)
img_xt_game = draw_top_xt_map(df_game, top_n=5)
img_cross_game = draw_cross_map(df_game_filtered)

# ---- Maps: Pass Map | Cross Map | Zone Heatmap | Top 5 Pass Impact ----
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown("**Pass Map**", unsafe_allow_html=True)
    st.image(img_pm_game, use_container_width=True)
with col_m2:
    st.markdown("**Cross Map**", unsafe_allow_html=True)
    st.image(img_cross_game, use_container_width=True)

col_m3, col_m4 = st.columns(2)
with col_m3:
    st.markdown("**Zone Heatmap**", unsafe_allow_html=True)
    st.image(img_ht_game, use_container_width=True)
with col_m4:
    st.markdown("**Top 5 Pass Impact**", unsafe_allow_html=True)
    st.image(img_xt_game, use_container_width=True)

# ---- Compact stats cards below maps ----
st.markdown("<br>", unsafe_allow_html=True)

if force_avg:
    c1, c2, c3 = st.columns(3)
    with c1:
        section_card("Pass Overview", C_BLUE_PASTEL, [
            ("Total Passes p90", f"{s_game['total_p90']:.1f}", f"Total: {int(s_game['total_passes'])}"),
        ])
    with c2:
        section_card("Progression", C_GREEN_PASTEL, [
            ("Progressive p90", f"{s_game['prog_p90']:.1f}", f"Total: {int(s_game['progressive_successful'])}"),
        ])
    with c3:
        section_card("Impact", C_AMBER_PASTEL, [
            ("Pass Impact Value", f"{s_game['xt_p90']:.3f}", f"Total xT: {s_game['sum_dxt']:.3f}"),
        ])
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        cmp_section_card("Pass Overview", C_BLUE_PASTEL, [
            ("Total Passes p90", s_game["total_p90"], f"{s_avg['total_p90']:.1f}",
             f"{s_game['total_p90']:.1f}", f"{s_avg['total_p90']:.1f}"),
        ])
    with c2:
        cmp_section_card("Progression", C_GREEN_PASTEL, [
            ("Progressive p90", s_game["prog_p90"], f"{s_avg['prog_p90']:.1f}"),
        ])
    with c3:
        cmp_section_card("Impact", C_AMBER_PASTEL, [
            ("Pass Impact Value", s_game["xt_p90"], s_avg["xt_p90"],
             f"{s_game['xt_p90']:.3f}", f"{s_avg['xt_p90']:.3f}"),
        ])

# ============================================================
# MATCH DETAILS — DEFENSIVE ACTIONS
# ============================================================
st.markdown("---")
st.markdown("### Match Details — Defensive Actions")

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

img_def_map = draw_defensive_map(df_def_game)

col_dm, col_ds_right = st.columns([3, 2])
with col_dm:
    st.markdown("**Defensive Actions Map**", unsafe_allow_html=True)
    st.image(img_def_map, use_container_width=True)
with col_ds_right:
    st.markdown("<br>", unsafe_allow_html=True)
    if force_avg_def:
        section_card("General", C_BLUE_PASTEL, [
            ("Defensive Actions p90", f"{d_game['total_actions_p90']:.1f}"),
            ("Actions in Opp. Field p90", f"{d_game['actions_attacking_p90']:.1f}"),
        ])
        section_card("Duels", C_GREEN_PASTEL, [
            ("Defensive Duels p90", f"{d_game['duels_p90']:.1f}"),
            ("% Duels Won", f"{d_game['duels_won_pct']:.1f}%"),
        ])
        section_card("Interceptions", C_AMBER_PASTEL, [
            ("Interceptions p90", f"{d_game['interceptions_p90']:.1f}"),
            ("Interceptions in Opp. Field p90", f"{d_game['interceptions_attacking_p90']:.1f}"),
        ])
    else:
        cmp_section_card("General", C_BLUE_PASTEL, [
            ("Defensive Actions p90", d_game["total_actions_p90"], f"{d_avg['total_actions_p90']:.1f}"),
            ("Actions in Opp. Field p90", d_game["actions_attacking_p90"], f"{d_avg['actions_attacking_p90']:.1f}"),
        ])
        cmp_section_card("Duels", C_GREEN_PASTEL, [
            ("Defensive Duels p90", d_game["duels_p90"], f"{d_avg['duels_p90']:.1f}"),
            ("% Duels Won", d_game["duels_won_pct"], d_avg["duels_won_pct"],
             f"{d_game['duels_won_pct']:.1f}%", f"{d_avg['duels_won_pct']:.1f}%"),
        ])
        cmp_section_card("Interceptions", C_AMBER_PASTEL, [
            ("Interceptions p90", d_game["interceptions_p90"], f"{d_avg['interceptions_p90']:.1f}"),
            ("Interceptions in Opp. Field p90", d_game["interceptions_attacking_p90"], f"{d_avg['interceptions_attacking_p90']:.1f}"),
        ])

st.markdown("<br>", unsafe_allow_html=True)
