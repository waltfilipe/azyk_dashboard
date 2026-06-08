import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from mplsoccer import Pitch
import math

st.set_page_config(layout='wide', page_title='Azyk Analytics')

PITCH_TYPE = 'statsbomb'
PITCH_COLOR = '#22312b'
LINE_COLOR = '#c7d5cc'

# ── MATCH DATA ──────────────────────────────────────────────

MATCH_DATA = {
    "vs Salt Lake": {
        "min": 85,
        "passes_s": [
            (115.52,72.91,109.54,73.08), (108.21,53.46,108.04,39.67),
            (103.05,66.10,96.24,50.81), (98.07,73.08,90.59,61.94),
            (92.25,67.10,74.63,53.13), (89.59,74.91,80.28,60.61),
            (69.31,53.46,72.97,64.94), (63.66,68.76,75.13,49.14),
            (67.65,63.44,67.65,51.14), (73.30,74.91,52.35,66.60),
            (55.35,64.44,49.53,48.64), (53.18,51.97,42.21,41.16),
            (42.71,61.28,41.55,38.50), (19.77,58.62,42.38,58.12),
            (23.93,70.92,8.30,52.14), (66.15,58.95,94.24,68.76),
            (97.90,69.59,88.76,66.76), (74.30,65.77,62.99,53.80),
            (69.48,69.59,60.66,64.44), (62.99,72.91,45.70,62.77),
            (51.19,69.59,51.02,54.63), (46.54,74.58,38.56,62.77),
            (39.39,66.10,49.36,60.78), (42.71,67.59,45.21,57.62),
            (47.86,61.28,35.23,50.14)
        ],
        "passes_f": [
            (44.21,63.27,69.48,19.72), (49.69,71.58,55.51,60.61),
            (36.06,55.96,23.26,56.29), (18.11,74.08,18.94,60.61)
        ],
        "interceptions": [(27.75,56.96), (24.26,59.95), (10.96,51.64)],
        "def_duels_s": [(109.70,68.92), (68.64,55.79), (19.44,78.40), (28.08,69.42)],
        "def_duels_f": [(2.65,73.75), (17.28,64.94), (29.25,58.62)],
        "off_duels_s": [],
        "off_duels_f": [(49.86,67.10)],
        "crosses_s": [(89.92,70.42,103.22,30.36)],
        "crosses_f": [(114.86,70.92,110.70,40.17), (105.88,72.42,111.53,62.61)]
    },
    "vs Vancouver": {
        "min": 45,
        "passes_s": [
            (106.55,57.45,118.02,57.12), (85.60,71.92,84.27,56.62),
            (97.57,65.10,88.09,54.13), (77.62,63.61,90.59,50.14),
            (66.32,56.62,79.95,46.15), (83.61,74.58,103.22,67.76),
            (74.96,59.28,95.57,75.57), (84.94,60.45,70.47,53.80),
            (70.81,66.76,64.65,51.80), (58.17,64.27,59.67,55.96),
            (56.01,68.76,58.17,55.46), (56.34,70.59,54.51,55.79),
            (55.35,73.91,81.94,64.27), (47.70,63.61,83.77,78.23),
            (48.70,68.59,49.03,54.13), (27.25,69.92,46.70,78.07),
            (36.39,69.09,59.34,61.28), (42.71,69.76,17.28,68.26),
            (25.26,67.26,32.24,53.63)
        ],
        "passes_f": [
            (38.89,63.27,48.03,62.61), (51.85,66.26,101.39,66.93),
            (62.99,63.61,90.59,78.90)
        ],
        "interceptions": [(75.96,63.61), (27.25,50.97), (35.23,66.76)],
        "def_duels_s": [(49.19,64.77)],
        "def_duels_f": [],
        "off_duels_s": [],
        "off_duels_f": [],
        "crosses_s": [],
        "crosses_f": []
    },
    "vs LAFC": {
        "min": 82,
        "passes_s": [
            (97.40,56.29,117.68,56.12), (99.40,75.91,93.58,66.76),
            (82.44,61.78,74.13,55.29), (75.79,59.28,87.76,72.08),
            (63.66,68.76,72.63,55.63), (46.37,57.29,66.82,77.57),
            (44.54,56.46,38.06,54.63), (27.25,60.45,34.57,49.31),
            (37.23,67.10,22.43,62.61), (33.40,70.75,24.92,61.28),
            (25.76,71.42,18.28,66.76), (20.27,76.24,35.90,73.91),
            (13.12,53.63,11.29,24.21)
        ],
        "passes_f": [(112.20,63.27,103.89,56.29)],
        "interceptions": [(41.22,60.45), (25.42,64.27), (25.59,67.76)],
        "def_duels_s": [(55.35,70.59), (31.74,65.77), (21.93,64.27)],
        "def_duels_f": [(24.92,53.13), (13.29,68.09), (5.48,72.91), (1.15,69.92)],
        "off_duels_s": [(112.53,64.77), (108.21,54.96)],
        "off_duels_f": [],
        "crosses_s": [],
        "crosses_f": [(99.90,61.94,107.54,36.68)]
    },
    "vs Portland": {
        "min": 0, "passes_s": [], "passes_f": [], "interceptions": [],
        "def_duels_s": [], "def_duels_f": [], "off_duels_s": [], "off_duels_f": [],
        "crosses_s": [], "crosses_f": []
    }
}

# ── HELPERS ──────────────────────────────────────────────

def calc_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def is_progressive(x1, y1, x2, y2):
    return (x2 - x1) > 15

def per_90(val, mins):
    return (val / mins * 90) if mins > 0 else 0

def calc_match_grade(pass_acc, def_success):
    return min(100, pass_acc * 0.75 + def_success * 0.25)

# ── UI ──────────────────────────────────────────────

st.title("⚽ Azyk Performance Dashboard")
match_name = st.sidebar.selectbox("Select Match", list(MATCH_DATA.keys()))
data = MATCH_DATA[match_name]

if data['min'] == 0:
    st.warning("🚫 Did Not Play — no data available")
    st.stop()

# Compute stats
total_passes = len(data['passes_s']) + len(data['passes_f'])
pass_acc = (len(data['passes_s']) / total_passes * 100) if total_passes > 0 else 0
total_def = len(data['def_duels_s']) + len(data['def_duels_f'])
def_rate = (len(data['def_duels_s']) / total_def * 100) if total_def > 0 else 0
prog = sum(1 for p in data['passes_s'] if is_progressive(*p))
ints = len(data['interceptions'])
grade = calc_match_grade(pass_acc, def_rate)

# Funnel
funnel = 0
for p in data['passes_s']:
    if 95 <= p[2] <= 120 and 25 <= p[3] <= 55:
        funnel += 1
for d in data['def_duels_s']:
    if 95 <= d[0] <= 120 and 25 <= d[1] <= 55:
        funnel += 1
for i in data['interceptions']:
    if 95 <= i[0] <= 120 and 25 <= i[1] <= 55:
        funnel += 1

# ── TABS ──────────────────────────────────────────────

tab1, tab2 = st.tabs(["📊 Charts", "📋 Dashboard"])

# ═══════════════════════════════════════════════════════
# TAB 1: CHARTS
# ═══════════════════════════════════════════════════════

with tab1:
    pitch = Pitch(pitch_type=PITCH_TYPE, pitch_color=PITCH_COLOR, line_color=LINE_COLOR)

    # ── 1. PASS MAP ──
    st.subheader("Pass Distribution")
    fig1, ax1 = pitch.draw(figsize=(10, 7))
    for p in data['passes_s']:
        ax1.annotate("", xy=(p[2], p[3]), xytext=(p[0], p[1]),
                      arrowprops=dict(arrowstyle="->", color="green", lw=1.5, alpha=0.7))
    for p in data['passes_f']:
        ax1.annotate("", xy=(p[2], p[3]), xytext=(p[0], p[1]),
                      arrowprops=dict(arrowstyle="->", color="red", lw=1.5, linestyle="dashed", alpha=0.7))
    ax1.set_title(f"Pass Distribution — {match_name}", color='white', fontsize=14)
    st.pyplot(fig1)
    plt.close(fig1)

    # ── 2. DEFENSIVE ACTIONS ──
    st.subheader("Defensive Actions & Interceptions")
    fig2, ax2 = pitch.draw(figsize=(10, 7))
    for d in data['def_duels_s']:
        ax2.scatter(d[0], d[1], s=120, c='#00ff88', edgecolors='white', linewidths=1.5, zorder=5)
    for d in data['def_duels_f']:
        ax2.scatter(d[0], d[1], s=120, c='#ff4444', edgecolors='white', linewidths=1.5, zorder=5)
    for i in data['interceptions']:
        ax2.scatter(i[0], i[1], s=180, marker='*', c='#ffdd00', edgecolors='black', linewidths=1, zorder=5)

    # Legend
    if data['def_duels_s']:
        ax2.scatter([], [], s=100, c='#00ff88', label='Def Duel Won')
    if data['def_duels_f']:
        ax2.scatter([], [], s=100, c='#ff4444', label='Def Duel Lost')
    if data['interceptions']:
        ax2.scatter([], [], s=150, marker='*', c='#ffdd00', label='Interception')
    ax2.legend(loc='lower left', fontsize=9, facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax2.set_title(f"Defensive Actions — {match_name}", color='white', fontsize=14)
    st.pyplot(fig2)
    plt.close(fig2)

    # ── 3. CROSS ANALYSIS ──
    st.subheader("Cross Analysis")
    fig3, ax3 = pitch.draw(figsize=(10, 7))
    if not data['crosses_s'] and not data['crosses_f']:
        ax3.text(60, 40, "No crosses in this match", ha='center', va='center',
                 fontsize=14, color='white', fontweight='bold')
    for c in data['crosses_s']:
        ax3.annotate("", xy=(c[2], c[3]), xytext=(c[0], c[1]),
                      arrowprops=dict(arrowstyle="->", color="cyan", lw=2.5))
    for c in data['crosses_f']:
        ax3.annotate("", xy=(c[2], c[3]), xytext=(c[0], c[1]),
                      arrowprops=dict(arrowstyle="->", color="red", lw=2.5, linestyle="dashed"))
    ax3.set_title(f"Cross Analysis — {match_name}", color='white', fontsize=14)
    st.pyplot(fig3)
    plt.close(fig3)

    # ── 4. HEATMAP ──
    st.subheader("Action Heatmap")
    all_x, all_y = [], []
    for p in data['passes_s']:
        all_x.extend([p[0], p[2]]); all_y.extend([p[1], p[3]])
    for p in data['passes_f']:
        all_x.extend([p[0], p[2]]); all_y.extend([p[1], p[3]])
    for d in data['def_duels_s']:
        all_x.append(d[0]); all_y.append(d[1])
    for d in data['def_duels_f']:
        all_x.append(d[0]); all_y.append(d[1])
    for i in data['interceptions']:
        all_x.append(i[0]); all_y.append(i[1])
    for o in data['off_duels_s']:
        all_x.append(o[0]); all_y.append(o[1])
    for o in data['off_duels_f']:
        all_x.append(o[0]); all_y.append(o[1])
    for c in data['crosses_s']:
        all_x.extend([c[0], c[2]]); all_y.extend([c[1], c[3]])
    for c in data['crosses_f']:
        all_x.extend([c[0], c[2]]); all_y.extend([c[1], c[3]])

    fig4, ax4 = pitch.draw(figsize=(10, 7))
    if all_x:
        H, xedges, yedges = np.histogram2d(all_x, all_y, bins=(12, 9), range=[[0, 120], [0, 80]])
        ax4.pcolormesh(xedges, yedges, H.T, cmap='hot', alpha=0.6, zorder=2)
    ax4.set_title(f"Action Heatmap — {match_name}", color='white', fontsize=14)
    st.pyplot(fig4)
    plt.close(fig4)

# ═══════════════════════════════════════════════════════
# TAB 2: DASHBOARD
# ═══════════════════════════════════════════════════════

with tab2:
    st.markdown(f"## Match Summary: {match_name}")
    st.markdown(f"**Minutes played:** {data['min']}")

    # ── 6 STAT CARDS ──
    CARD = (
        "background:#1a1a2e; color:white; padding:20px; border-radius:12px; "
        "text-align:center; box-shadow:0 4px 6px rgba(0,0,0,0.3);"
    )
    VAL = "color:#00ffcc; font-size:32px; font-weight:bold; margin:8px 0 0 0;"
    LAB = "font-size:14px; color:#aaa; margin:0;"

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div style='{CARD}'><p style='{LAB}'>🎯 Pass Accuracy</p>"
        f"<p style='{VAL}'>{pass_acc:.1f}%</p></div>",
        unsafe_allow_html=True
    )
    col2.markdown(
        f"<div style='{CARD}'><p style='{LAB}'>📤 Total Passes</p>"
        f"<p style='{VAL}'>{total_passes}</p></div>",
        unsafe_allow_html=True
    )
    col3.markdown(
        f"<div style='{CARD}'><p style='{LAB}'>📈 Progressive Passes</p>"
        f"<p style='{VAL}'>{prog}</p></div>",
        unsafe_allow_html=True
    )

    col4, col5, col6 = st.columns(3)
    col4.markdown(
        f"<div style='{CARD}'><p style='{LAB}'>🛡️ Def Duels Won</p>"
        f"<p style='{VAL}'>{def_rate:.1f}%</p></div>",
        unsafe_allow_html=True
    )
    col5.markdown(
        f"<div style='{CARD}'><p style='{LAB}'>👁️ Interceptions</p>"
        f"<p style='{VAL}'>{ints}</p></div>",
        unsafe_allow_html=True
    )
    col6.markdown(
        f"<div style='{CARD}'><p style='{LAB}'>⭐ Match Grade</p>"
        f"<p style='{VAL}'>{grade:.1f}</p></div>",
        unsafe_allow_html=True
    )

    # ── MATCH GRADE PROGRESS ──
    st.markdown("### Match Grade")
    st.progress(min(grade / 100, 1.0))
    if grade >= 80:
        st.success(f"🌟 Excellent performance! ({grade:.1f}/100)")
    elif grade >= 60:
        st.info(f"✅ Solid performance ({grade:.1f}/100)")
    else:
        st.warning(f"⚠️ Room for improvement ({grade:.1f}/100)")

    # ── PER 90 ──
    st.markdown("### Per 90 Minutes")
    a, b, c, d = st.columns(4)
    a.metric("Passes p90", f"{per_90(total_passes, data['min']):.1f}")
    b.metric("Progressive p90", f"{per_90(prog, data['min']):.1f}")
    c.metric("Def Duels p90", f"{per_90(total_def, data['min']):.1f}")
    d.metric("Interceptions p90", f"{per_90(ints, data['min']):.1f}")

    # ── THE FUNNEL ──
    st.markdown("### 🔥 The Funnel (Attacking Third Central)")
    st.info(
        f"**{funnel} events** reached the high-danger central attacking zone "
        f"(x: 95–120, y: 25–55) — the area right in front of goal."
    )

    # ── DETAILED STATS ──
    st.markdown("### Detailed Stats")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Passes Success", len(data['passes_s']))
    e2.metric("Passes Failed", len(data['passes_f']))
    e3.metric("Def Duels Won", len(data['def_duels_s']))
    e4.metric("Def Duels Lost", len(data['def_duels_f']))

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Off Duels Won", len(data['off_duels_s']))
    f2.metric("Off Duels Lost", len(data['off_duels_f']))
    f3.metric("Crosses Success", len(data['crosses_s']))
    f4.metric("Crosses Failed", len(data['crosses_f']))

# ═══════════════════════════════════════════════════════
# COMPARATIVE ANALYSIS (outside tabs)
# ═══════════════════════════════════════════════════════

st.markdown("---")
st.subheader("📊 Comparative Analysis Across Matches")

played = [m for m in MATCH_DATA if MATCH_DATA[m]['min'] > 0]
comp = []
for m in played:
    d = MATCH_DATA[m]
    tp = len(d['passes_s']) + len(d['passes_f'])
    td = len(d['def_duels_s']) + len(d['def_duels_f'])
    pa = (len(d['passes_s']) / tp * 100) if tp > 0 else 0
    dr = (len(d['def_duels_s']) / td * 100) if td > 0 else 0
    comp.append({
        'Match': m,
        'Pass Acc': round(pa, 1),
        'Def Duel %': round(dr, 1),
        'Grade': round(calc_match_grade(pa, dr), 1),
        'Prog': sum(1 for p in d['passes_s'] if is_progressive(*p)),
        'Ints': len(d['interceptions'])
    })
df = pd.DataFrame(comp)

# ── Grouped Bar: Pass Acc vs Grade ──
fig5 = go.Figure()
fig5.add_trace(go.Bar(name='Pass Accuracy %', x=df['Match'], y=df['Pass Acc'],
                       marker_color='#00cc88'))
fig5.add_trace(go.Bar(name='Match Grade', x=df['Match'], y=df['Grade'],
                       marker_color='#ffaa00'))
fig5.update_layout(
    title='Pass Accuracy vs Match Grade',
    barmode='group',
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=400
)
st.plotly_chart(fig5, use_container_width=True)

# ── Grouped Bar: Defensive ──
fig6 = go.Figure()
fig6.add_trace(go.Bar(name='Def Duel %', x=df['Match'], y=df['Def Duel %'],
                       marker_color='#4488ff'))
fig6.add_trace(go.Bar(name='Interceptions', x=df['Match'], y=df['Ints'],
                       marker_color='#ffdd00'))
fig6.update_layout(
    title='Defensive Performance',
    barmode='group',
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=400
)
st.plotly_chart(fig6, use_container_width=True)

# ── Progressive Passes ──
fig7 = px.bar(
    df, x='Match', y='Prog', title='Progressive Passes per Match',
    color='Prog', color_continuous_scale='viridis', template='plotly_dark',
    height=400
)
fig7.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig7, use_container_width=True)

# ── Radar Chart ──
fig8 = go.Figure()
for _, row in df.iterrows():
    fig8.add_trace(go.Scatterpolar(
        r=[row['Pass Acc'], row['Def Duel %'], row['Prog'] * 10,
           row['Ints'] * 10, row['Grade']],
        theta=['Pass Acc', 'Def Duel %', 'Prog (×10)', 'Ints (×10)', 'Grade'],
        fill='toself',
        name=row['Match']
    ))
fig8.update_layout(
    title='Performance Radar',
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    template='plotly_dark',
    height=500,
    paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig8, use_container_width=True)

# ── Summary Table ──
st.markdown("### Summary Table")
st.dataframe(df.set_index('Match'), use_container_width=True)
