import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from mplsoccer import Pitch
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Azyk Performance Dashboard")

XT_GRID = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.01, 0.02, 0.03, 0.04, 0.03, 0.02],
    [0.02, 0.04, 0.06, 0.08, 0.06, 0.04],
    [0.03, 0.06, 0.10, 0.12, 0.10, 0.06],
    [0.04, 0.08, 0.15, 0.20, 0.15, 0.08]
])

MATCHES = {
    "SALT LAKE": {
        "min": 85,
        "passes_s": [
            (115.52, 72.91, 109.54, 73.08), (108.21, 53.46, 108.04, 39.67),
            (103.05, 66.10, 96.24, 50.81), (98.07, 73.08, 90.59, 61.94),
            (92.25, 67.10, 74.63, 53.13), (89.59, 74.91, 80.28, 60.61),
            (69.31, 53.46, 72.97, 64.94), (63.66, 68.76, 75.13, 49.14),
            (67.65, 63.44, 67.65, 51.14), (73.30, 74.91, 52.35, 66.60),
            (55.35, 64.44, 49.53, 48.64), (53.18, 51.97, 42.21, 41.16),
            (42.71, 61.28, 41.55, 38.50), (19.77, 58.62, 42.38, 58.12),
            (23.93, 70.92, 8.30, 52.14), (66.15, 58.95, 94.24, 68.76),
            (97.90, 69.59, 88.76, 66.76), (74.30, 65.77, 62.99, 53.80),
            (69.48, 69.59, 60.66, 64.44), (62.99, 72.91, 45.70, 62.77),
            (51.19, 69.59, 51.02, 54.63), (46.54, 74.58, 38.56, 62.77),
            (39.39, 66.10, 49.36, 60.78), (42.71, 67.59, 45.21, 57.62),
            (47.86, 61.28, 35.23, 50.14)
        ],
        "passes_f": [
            (44.21, 63.27, 69.48, 19.72), (49.69, 71.58, 55.51, 60.61),
            (36.06, 55.96, 23.26, 56.29), (18.11, 74.08, 18.94, 60.61)
        ],
        "interc": [(27.75, 56.96), (24.26, 59.95), (10.96, 51.64)],
        "duels_s": [(109.70, 68.92), (68.64, 55.79), (19.44, 78.40), (28.08, 69.42)],
        "duels_f": [(2.65, 73.75), (17.28, 64.94), (29.25, 58.62)],
        "duels_of": [(49.86, 67.10)],
        "cross_s": [(89.92, 70.42, 103.22, 30.36)],
        "cross_f": [(114.86, 70.92, 110.70, 40.17), (105.88, 72.42, 111.53, 62.61)]
    },
    "VANCOUVER": {
        "min": 45,
        "passes_s": [
            (106.55, 57.45, 118.02, 57.12), (85.60, 71.92, 84.27, 56.62),
            (97.57, 65.10, 88.09, 54.13), (77.62, 63.61, 90.59, 50.14),
            (66.32, 56.62, 79.95, 46.15), (83.61, 74.58, 103.22, 67.76),
            (74.96, 59.28, 95.57, 75.57), (84.94, 60.45, 70.47, 53.80),
            (70.81, 66.76, 64.65, 51.80), (58.17, 64.27, 59.67, 55.96),
            (56.01, 68.76, 58.17, 55.46), (56.34, 70.59, 54.51, 55.79),
            (55.35, 73.91, 81.94, 64.27), (47.70, 63.61, 83.77, 78.23),
            (48.70, 68.59, 49.03, 54.13), (27.25, 69.92, 46.70, 78.07),
            (36.39, 69.09, 59.34, 61.28), (42.71, 69.76, 17.28, 68.26),
            (25.26, 67.26, 32.24, 53.63)
        ],
        "passes_f": [
            (38.89, 63.27, 48.03, 62.61), (51.85, 66.26, 101.39, 66.93),
            (62.99, 63.61, 90.59, 78.90)
        ],
        "interc": [(75.96, 63.61), (27.25, 50.97), (35.23, 66.76)],
        "duels_s": [(49.19, 64.77)],
        "duels_f": [],
        "duels_of": [],
        "cross_s": [],
        "cross_f": []
    },
    "LAFC": {
        "min": 82,
        "passes_s": [
            (97.40, 56.29, 117.68, 56.12), (99.40, 75.91, 93.58, 66.76),
            (82.44, 61.78, 74.13, 55.29), (75.79, 59.28, 87.76, 72.08),
            (63.66, 68.76, 72.63, 55.63), (46.37, 57.29, 66.82, 77.57),
            (44.54, 56.46, 38.06, 54.63), (27.25, 60.45, 34.57, 49.31),
            (37.23, 67.10, 22.43, 62.61), (33.40, 70.75, 24.92, 61.28),
            (25.76, 71.42, 18.28, 66.76), (20.27, 76.24, 35.90, 73.91),
            (13.12, 53.63, 11.29, 24.21)
        ],
        "passes_f": [(112.20, 63.27, 103.89, 56.29)],
        "interc": [(41.22, 60.45), (25.42, 64.27), (25.59, 67.76)],
        "duels_s": [(55.35, 70.59), (31.74, 65.77), (21.93, 64.27)],
        "duels_f": [(24.92, 53.13), (13.29, 68.09), (5.48, 72.91), (1.15, 69.92)],
        "duels_of": [(112.53, 64.77), (108.21, 54.96)],
        "cross_s": [],
        "cross_f": [(99.90, 61.94, 107.54, 36.68)]
    }
}

def p90(v, m):
    return (v / m) * 90 if m > 0 else 0

def is_prog(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) > 15 or x2 > 80

def is_f3(x2):
    return x2 > 80

def is_opp(x):
    return x > 60

def is_funnel(x, y):
    return x < 30 and 18 <= y <= 62

def get_xt(x, y):
    r = min(int(y / 16), 4)
    c = min(int(x / 20), 5)
    return XT_GRID[r][c]

def match_stats(d):
    ps = d['passes_s']
    pf = d['passes_f']
    mins = d['min']
    total = len(ps) + len(pf)
    acc = (len(ps) / total * 100) if total > 0 else 0
    prog = sum(1 for p in ps if is_prog(p[0], p[1], p[2], p[3]))
    f3 = sum(1 for p in ps if is_f3(p[2]))
    xt = sum(get_xt(p[2], p[3]) for p in ps)
    ds = len(d['duels_s'])
    df = len(d['duels_f'])
    dt = ds + df
    d_win = (ds / dt * 100) if dt > 0 else 0
    interc = len(d['interc'])
    do = len(d['duels_of'])
    cs = len(d['cross_s'])
    cf = len(d['cross_f'])
    def_act = dt + interc
    off_score = prog * 0.3 + f3 * 0.2 + acc / 10 * 0.2 + cs * 0.15 + do * 0.1
    def_score = ds * 0.4 + interc * 0.3 + def_act * 0.3
    grade = min(10, (off_score * 0.75 + def_score * 0.25) / 3)
    return {
        'total_passes': total, 'acc': round(acc, 1), 'prog': prog, 'f3': f3,
        'xt': round(xt, 3), 'duels': dt, 'd_win': round(d_win, 1),
        'interc': interc, 'off_duels': do, 'crosses': cs + cf,
        'def_act': def_act, 'grade': round(grade, 1),
        'ps': len(ps), 'pf': len(pf), 'ds': ds, 'dt': dt, 'mins': mins
    }

all_stats = {k: match_stats(v) for k, v in MATCHES.items()}

def card(label, value, sub="", color="#e94560"):
    return (
        f"<div style='background:#16213e;border-radius:10px;padding:12px;"
        f"text-align:center;border:1px solid #0f3460;margin:5px;'>"
        f"<div style='color:#8892b0;font-size:11px;'>{label}</div>"
        f"<div style='color:{color};font-size:22px;font-weight:bold;'>{value}</div>"
        f"<div style='color:#4ecca3;font-size:10px;'>{sub}</div></div>"
    )

with st.sidebar:
    st.markdown(
        "<h1 style='color:#e94560;text-align:center;font-size:42px;'>AZYK</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;color:#8892b0;'>⭐ 2026 Season</p>",
        unsafe_allow_html=True
    )
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", len(MATCHES))
    total_min = sum(m['min'] for m in MATCHES.values())
    c2.metric("Min", total_min)
    c3.metric("Avg Min", f"{total_min / len(MATCHES):.0f}")

tab1, tab2, tab3 = st.tabs(["📊 Charts & Analysis", "📋 Detailed Dashboard", "📈 Development"])

with tab1:
    avg_mins = np.mean([m['min'] for m in MATCHES.values()])

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📋 Overview — Passes")
        cols = st.columns(3)
        cols[0].markdown(
            card('Passes p90',
                 f"{p90(np.mean([s['total_passes'] for s in all_stats.values()]), avg_mins):.1f}",
                 f"Total: {sum(s['total_passes'] for s in all_stats.values())}"),
            unsafe_allow_html=True
        )
        cols[1].markdown(
            card('Successful %',
                 f"{np.mean([s['acc'] for s in all_stats.values()]):.1f}%",
                 'Accuracy Rate', '#4ecca3'),
            unsafe_allow_html=True
        )
        cols[2].markdown(
            card('Total Passes',
                 f"{sum(s['total_passes'] for s in all_stats.values())}",
                 '3 matches'),
            unsafe_allow_html=True
        )

        st.markdown("### 📊 Advanced")
        cols = st.columns(3)
        cols[0].markdown(
            card('Progressive p90',
                 f"{p90(np.mean([s['prog'] for s in all_stats.values()]), avg_mins):.1f}",
                 f"Total: {sum(s['prog'] for s in all_stats.values())}"),
            unsafe_allow_html=True
        )
        cols[1].markdown(
            card('Final Third p90',
                 f"{p90(np.mean([s['f3'] for s in all_stats.values()]), avg_mins):.1f}",
                 f"Total: {sum(s['f3'] for s in all_stats.values())}"),
            unsafe_allow_html=True
        )
        cols[2].markdown(
            card('xT Total',
                 f"{sum(s['xt'] for s in all_stats.values()):.3f}",
                 'Expected Threat'),
            unsafe_allow_html=True
        )

        st.markdown("### ⚡ Impact")
        cols = st.columns(3)
        pct_prog = np.mean([s['prog'] / max(s['total_passes'], 1) * 100 for s in all_stats.values()])
        cols[0].markdown(
            card('% Progressive', f"{pct_prog:.1f}%", 'Positive Impact', '#faca15'),
            unsafe_allow_html=True
        )
        avg_dist = np.mean([
            np.mean([np.sqrt((p[2] - p[0]) ** 2 + (p[3] - p[1]) ** 2) for p in m['passes_s']])
            for m in MATCHES.values()
        ])
        cols[1].markdown(
            card('Avg Distance', f"{avg_dist:.1f}m", 'Successful Passes'),
            unsafe_allow_html=True
        )
        xt_per = sum(s['xt'] for s in all_stats.values()) / max(
            sum(s['total_passes'] for s in all_stats.values()), 1
        )
        cols[2].markdown(
            card('xT per Pass', f"{xt_per:.3f}", 'Avg Threat Added'),
            unsafe_allow_html=True
        )

    with c2:
        st.markdown("### 🛡️ General — Defense")
        cols = st.columns(3)
        avg_def = np.mean([s['interc'] + s['duels'] for s in all_stats.values()])
        cols[0].markdown(
            card('Def Actions p90',
                 f"{p90(avg_def, avg_mins):.1f}",
                 f"Total: {sum(s['interc'] + s['duels'] for s in all_stats.values())}"),
            unsafe_allow_html=True
        )
        opp_acts = np.mean([
            sum(1 for p in MATCHES[m]['interc'] if is_opp(p[0])) +
            sum(1 for p in MATCHES[m]['duels_s'] + MATCHES[m]['duels_f'] if is_opp(p[0]))
            for m in MATCHES
        ])
        cols[1].markdown(
            card('Opp Field p90', f"{p90(opp_acts, avg_mins):.1f}", 'x > 60', '#0f3460'),
            unsafe_allow_html=True
        )
        funnel = np.mean([
            sum(1 for p in MATCHES[m]['interc'] if is_funnel(*p)) +
            sum(1 for p in MATCHES[m]['duels_s'] + MATCHES[m]['duels_f'] if is_funnel(*p))
            for m in MATCHES
        ])
        cols[2].markdown(
            card('Funnel Zone p90', f"{p90(funnel, avg_mins):.1f}", 'Area +15m'),
            unsafe_allow_html=True
        )

        st.markdown("### ⚔️ Duels")
        cols = st.columns(3)
        cols[0].markdown(
            card('Def Duels p90',
                 f"{p90(np.mean([s['duels'] for s in all_stats.values()]), avg_mins):.1f}",
                 f"Total: {sum(s['duels'] for s in all_stats.values())}"),
            unsafe_allow_html=True
        )
        cols[1].markdown(
            card('% Duels Won',
                 f"{np.mean([s['d_win'] for s in all_stats.values()]):.1f}%",
                 'Success Rate', '#4ecca3'),
            unsafe_allow_html=True
        )
        cols[2].markdown(
            card('Off Duels p90',
                 f"{p90(np.mean([s['off_duels'] for s in all_stats.values()]), avg_mins):.1f}",
                 f"Total: {sum(s['off_duels'] for s in all_stats.values())}"),
            unsafe_allow_html=True
        )

        st.markdown("### ❌ Interceptions")
        cols = st.columns(3)
        cols[0].markdown(
            card('Interceptions p90',
                 f"{p90(np.mean([s['interc'] for s in all_stats.values()]), avg_mins):.1f}",
                 f"Total: {sum(s['interc'] for s in all_stats.values())}"),
            unsafe_allow_html=True
        )
        int_opp = np.mean([
            sum(1 for p in MATCHES[m]['interc'] if is_opp(p[0])) for m in MATCHES
        ])
        cols[1].markdown(
            card('Interc Opp Field p90', f"{p90(int_opp, avg_mins):.1f}", 'x > 60', '#0f3460'),
            unsafe_allow_html=True
        )
        int_own = np.mean([
            sum(1 for p in MATCHES[m]['interc'] if not is_opp(p[0])) for m in MATCHES
        ])
        cols[2].markdown(
            card('Interc Own Half p90', f"{p90(int_own, avg_mins):.1f}", 'x <= 60'),
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("📊 Grade por Partida")
    gcols = st.columns(3)
    for i, name in enumerate(MATCHES.keys()):
        g = all_stats[name]['grade']
        gcolor = "#4ecca3" if g >= 7 else ("#faca15" if g >= 5 else "#e94560")
        gcols[i].markdown(
            f"<div style='background:#16213e;border-radius:15px;padding:20px;"
            f"text-align:center;border:2px solid {gcolor};margin:5px;'>"
            f"<div style='color:#8892b0;font-size:14px;'>{name}</div>"
            f"<div style='color:{gcolor};font-size:48px;font-weight:bold;'>{g:.1f}</div>"
            f"<div style='color:#8892b0;font-size:12px;'>{MATCHES[name]['min']} min</div></div>",
            unsafe_allow_html=True
        )

with tab2:
    col_f1, col_f2 = st.columns([1.5, 1])

    with col_f1:
        m_sel = st.selectbox("Select Match", list(MATCHES.keys()), key="tab2_match")
        pass_type = st.radio(
            "Pass Type",
            ["All", "Successful", "Unsuccessful", "Progressive", "Final Third"],
            horizontal=True
        )
        filter_def = st.radio(
            "Filter Type",
            ["All", "Duels Only", "Interceptions Only"],
            horizontal=True
        )

        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b',
                      line_color='#c7d5cc', linewidth=2)
        fig, ax = pitch.draw(figsize=(10, 7))

        d = MATCHES[m_sel]

        ps_all = list(d['passes_s']) + list(d['passes_f'])
        ps_colors = ['green'] * len(d['passes_s']) + ['red'] * len(d['passes_f'])

        if pass_type == "Successful":
            ps_all = list(d['passes_s'])
            ps_colors = ['green'] * len(d['passes_s'])
        elif pass_type == "Unsuccessful":
            ps_all = list(d['passes_f'])
            ps_colors = ['red'] * len(d['passes_f'])
        elif pass_type == "Progressive":
            ps_all = [p for p in d['passes_s'] if is_prog(p[0], p[1], p[2], p[3])]
            ps_colors = ['lime'] * len(ps_all)
        elif pass_type == "Final Third":
            ps_all = [p for p in d['passes_s'] if is_f3(p[2])]
            ps_colors = ['cyan'] * len(ps_all)

        for p, c in zip(ps_all, ps_colors):
            pitch.arrows(p[0], p[1], p[2], p[3], color=c, width=2,
                         headwidth=5, headlength=5, ax=ax)

        if filter_def != "Interceptions Only":
            for p in d['duels_s']:
                pitch.scatter(p[0], p[1], color='blue', marker='s', s=80, ax=ax)
            for p in d['duels_f']:
                pitch.scatter(p[0], p[1], color='red', marker='x', s=100, ax=ax)
            for p in d['duels_of']:
                pitch.scatter(p[0], p[1], color='orange', marker='o', s=80, ax=ax)

        if filter_def != "Duels Only":
            for p in d['interc']:
                pitch.scatter(p[0], p[1], color='yellow', marker='*', s=120, ax=ax)

        for p in d['cross_s']:
            pitch.arrows(p[0], p[1], p[2], p[3], color='purple', width=2, linestyle='--', ax=ax)
        for p in d['cross_f']:
            pitch.arrows(p[0], p[1], p[2], p[3], color='magenta', width=2, linestyle=':', ax=ax)

        st.pyplot(fig)

    with col_f2:
        s = all_stats[m_sel]
        st.markdown(f"<h3 style='color:#e94560;'>{m_sel}</h3>", unsafe_allow_html=True)

        st.markdown("**Passing**")
        cols = st.columns(2)
        cols[0].markdown(
            card('Total Passes', str(s['total_passes']), f"{s['ps']}S / {s['pf']}F"),
            unsafe_allow_html=True
        )
        cols[1].markdown(
            card('Accuracy', f"{s['acc']:.1f}%", '', '#4ecca3'),
            unsafe_allow_html=True
        )
        cols = st.columns(2)
        cols[0].markdown(
            card('Progressive', str(s['prog']), 'Positive Impact'),
            unsafe_allow_html=True
        )
        cols[1].markdown(
            card('Final Third', str(s['f3']), 'x > 80'),
            unsafe_allow_html=True
        )

        st.markdown("**Defensive**")
        cols = st.columns(2)
        cols[0].markdown(
            card('Def Duels', f"{s['ds']}/{s['dt']}", f"{s['d_win']:.0f}% Won"),
            unsafe_allow_html=True
        )
        cols[1].markdown(
            card('Interceptions', str(s['interc']), '', '#0f3460'),
            unsafe_allow_html=True
        )
        cols = st.columns(2)
        cols[0].markdown(
            card('Off Duels', str(s['off_duels']), '', '#faca15'),
            unsafe_allow_html=True
        )
        cols[1].markdown(
            card('Crosses', str(s['crosses']), ''),
            unsafe_allow_html=True
        )

        st.markdown("**Advanced**")
        cols = st.columns(2)
        cols[0].markdown(
            card('xT', f"{s['xt']:.3f}", 'Expected Threat'),
            unsafe_allow_html=True
        )
        gcolor = "#4ecca3" if s['grade'] >= 7 else ("#faca15" if s['grade'] >= 5 else "#e94560")
        cols[1].markdown(
            card('Grade', f"{s['grade']:.1f}", '/10', gcolor),
            unsafe_allow_html=True
        )

with tab3:
    st.subheader("📈 Performance Evolution")
    names = list(MATCHES.keys())
    grades_l = [all_stats[n]['grade'] for n in names]
    accs_l = [all_stats[n]['acc'] for n in names]
    defs_l = [all_stats[n]['def_act'] for n in names]
    progs_l = [all_stats[n]['prog'] for n in names]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=names, y=grades_l, mode='lines+markers', name='Grade',
        line=dict(color='#e94560', width=3), marker=dict(size=10)
    ))
    fig.add_trace(go.Scatter(
        x=names, y=accs_l, mode='lines+markers', name='Pass Acc %',
        line=dict(color='#4ecca3', width=2), marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=names, y=defs_l, mode='lines+markers', name='Def Actions',
        line=dict(color='#0f3460', width=2), marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=names, y=progs_l, mode='lines+markers', name='Progressive',
        line=dict(color='#faca15', width=2), marker=dict(size=8)
    ))
    fig.update_layout(
        template='plotly_dark', height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    x=0.5, xanchor='center')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Multi-Match Radar")
    categories = ['Passes', 'Accuracy', 'Progressive', 'Def Duels', 'Interceptions', 'Grade']
    fig_r = go.Figure()
    for name in names:
        s = all_stats[name]
        vals = [s['total_passes'], s['acc'], s['prog'], s['dt'], s['interc'], s['grade'] * 2]
        fig_r.add_trace(go.Scatterpolar(
            r=vals, theta=categories, fill='toself', name=name
        ))
    fig_r.update_layout(
        template='plotly_dark', height=400,
        polar=dict(radialaxis=dict(
            visible=True,
            range=[0, max(60, max([all_stats[n]['total_passes'] for n in names]) + 10)]
        ))
    )
    st.plotly_chart(fig_r, use_container_width=True)

    st.subheader("📋 Progression Table")
    theaders = ['Match', 'Min', 'Passes', 'Pass%', 'Prog', 'F3',
                'DefD', 'Duel%', 'Interc', 'OffD', 'Cross', 'Grade']
    table = (
        "<table style='width:100%;border-collapse:collapse;text-align:center;font-size:14px;'>"
        "<thead><tr style='background:#16213e;color:#e94560;'>"
    )
    for h in theaders:
        table += f"<th style='padding:8px;border:1px solid #0f3460;'>{h}</th>"
    table += "</tr></thead><tbody>"

    for name in names:
        s = all_stats[name]
        row = [
            name, str(s['mins']), str(s['total_passes']), f"{s['acc']:.1f}%",
            str(s['prog']), str(s['f3']), f"{s['ds']}/{s['dt']}",
            f"{s['d_win']:.0f}%", str(s['interc']), str(s['off_duels']),
            str(s['crosses']), f"{s['grade']:.1f}"
        ]
        table += "<tr style='color:white;'>"
        for r in row:
            table += f"<td style='padding:6px;border:1px solid #0f3460;'>{r}</td>"
        table += "</tr>"

    avg_m = np.mean([all_stats[n]['mins'] for n in names])
    avg_row = [
        'AVERAGE', f"{avg_m:.0f}",
        str(int(np.mean([all_stats[n]['total_passes'] for n in names]))),
        f"{np.mean([all_stats[n]['acc'] for n in names]):.1f}%",
        str(int(np.mean([all_stats[n]['prog'] for n in names]))),
        str(int(np.mean([all_stats[n]['f3'] for n in names]))),
        f"{int(np.mean([all_stats[n]['ds'] for n in names]))}/{int(np.mean([all_stats[n]['dt'] for n in names]))}",
        f"{np.mean([all_stats[n]['d_win'] for n in names]):.0f}%",
        str(int(np.mean([all_stats[n]['interc'] for n in names]))),
        str(int(np.mean([all_stats[n]['off_duels'] for n in names]))),
        str(int(np.mean([all_stats[n]['crosses'] for n in names]))),
        f"{np.mean([all_stats[n]['grade'] for n in names]):.1f}"
    ]
    table += (
        f"<tr style='background:#0f3460;color:#4ecca3;font-weight:bold;'>"
    )
    for r in avg_row:
        table += f"<td style='padding:6px;border:1px solid #0f3460;'>{r}</td>"
    table += "</tr></tbody></table>"

    st.markdown(table, unsafe_allow_html=True)
