import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", page_title="Football Performance Dashboard")

# ── Data ──
data = [
    {"match": "vs Salt Lake", "min": 85, "passes_success": 16, "passes_fail": 4, "interceptions": 3, "duel_def_s": 4, "duel_def_f": 3, "duel_off_s": 0, "duel_off_f": 1, "cross_s": 1, "cross_f": 2},
    {"match": "vs Vancouver", "min": 45, "passes_success": 19, "passes_fail": 3, "interceptions": 3, "duel_def_s": 1, "duel_def_f": 0, "duel_off_s": 0, "duel_off_f": 0, "cross_s": 0, "cross_f": 0},
    {"match": "vs LAFC",       "min": 82, "passes_success": 13, "passes_fail": 1, "interceptions": 3, "duel_def_s": 3, "duel_def_f": 4, "duel_off_s": 2, "duel_off_f": 0, "cross_s": 0, "cross_f": 1}
]
df = pd.DataFrame(data)

# ── CSS ──
st.markdown("""
<style>
    .main, .stApp { background: #0A0E17; }
    .block-container { padding-top: 2rem; }

    .card {
        height: 178px;
        border-radius: 14px;
        padding: 20px 22px 16px 22px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.35);
        border: 0.5px solid;
        margin-bottom: 24px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .card:hover { transform: translateY(-2px); box-shadow: 0 10px 32px rgba(0,0,0,0.45); }

    .card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        border-radius: 14px;
        background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, transparent 100%);
        pointer-events: none;
    }

    .blue-card  { background: linear-gradient(135deg, #1a2744 0%, #162040 100%); border-color: rgba(45, 74, 122, 0.25); }
    .green-card { background: linear-gradient(135deg, #1a3a2a 0%, #162e22 100%); border-color: rgba(45, 90, 58, 0.25); }
    .gold-card  { background: linear-gradient(135deg, #3a2e14 0%, #2e2410 100%); border-color: rgba(90, 74, 26, 0.25); }

    .card-header {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        color: #8892a4;
        opacity: 0.55;
        margin-bottom: 0;
        line-height: 1;
    }
    .card-label {
        font-size: 13px;
        color: #8892a4;
        margin: 0;
        line-height: 1.2;
    }
    .card-value {
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        margin: 2px 0 0 0;
        line-height: 1.1;
    }
    .card-value-small {
        font-size: 24px;
        font-weight: 400;
        color: #FFFFFF;
        opacity: 0.7;
        margin: 2px 0 0 0;
        line-height: 1.1;
    }
    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 6px;
        padding-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    .card-footer-item {
        font-size: 12px;
        color: #5a6478;
        letter-spacing: 0.3px;
    }
    .card-footer-item strong {
        color: #b0b8c8;
        font-weight: 600;
    }

    /* section title */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #e0e4ee;
        margin: 28px 0 14px 0;
        letter-spacing: 0.3px;
    }

    /* sidebar */
    .css-1d391kg, .css-12oz5g7 { background: #0D121E; }
    .stSelectbox label { color: #8892a4; font-size: 13px; }

    /* dataframe */
    .stDataFrame { background: transparent; }
    .stDataFrame td, .stDataFrame th { color: #c8ccd6 !important; background: transparent !important; border-color: rgba(255,255,255,0.06) !important; }
    .stDataFrame thead tr th { background: rgba(255,255,255,0.04) !important; font-weight: 600 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.5px; }

    /* divider */
    hr { border-color: rgba(255,255,255,0.06) !important; margin: 32px 0 !important; }

    /* download button */
    .stDownloadButton button {
        background: rgba(255,255,255,0.06) !important;
        color: #c8ccd6 !important;
        border: 0.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        transition: background 0.15s;
    }
    .stDownloadButton button:hover { background: rgba(255,255,255,0.12) !important; color: #fff !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚽ Scout Dashboard")
    st.markdown("---")
    match_select = st.selectbox("Partida", ["Todos os jogos"] + df["match"].tolist())

if match_select != "Todos os jogos":
    fdf = df[df["match"] == match_select].copy()
else:
    fdf = df.copy()

total_min = fdf["min"].sum()
factor = 90 / total_min if total_min > 0 else 0

# ── Helpers ──
def fmt_p90(v): return f"{v * factor:.1f}"

def card_blue(header, label, value, footer_left, footer_right):
    st.markdown(f"""
    <div class='card blue-card'>
        <div class='card-header'>{header}</div>
        <div>
            <div class='card-label'>{label}</div>
            <div class='card-value'>{value}</div>
        </div>
        <div class='card-footer'>
            <div class='card-footer-item'>{footer_left}</div>
            <div class='card-footer-item'>{footer_right}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def card_green(header, label, value, footer_left, footer_right):
    st.markdown(f"""
    <div class='card green-card'>
        <div class='card-header'>{header}</div>
        <div>
            <div class='card-label'>{label}</div>
            <div class='card-value'>{value}</div>
        </div>
        <div class='card-footer'>
            <div class='card-footer-item'>{footer_left}</div>
            <div class='card-footer-item'>{footer_right}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def card_gold(header, label, value, footer_left, footer_right):
    st.markdown(f"""
    <div class='card gold-card'>
        <div class='card-header'>{header}</div>
        <div>
            <div class='card-label'>{label}</div>
            <div class='card-value'>{value}</div>
        </div>
        <div class='card-footer'>
            <div class='card-footer-item'>{footer_left}</div>
            <div class='card-footer-item'>{footer_right}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Computed totals ──
p_s  = fdf["passes_success"].sum(); p_f  = fdf["passes_fail"].sum()
d_s  = fdf["duel_def_s"].sum() + fdf["duel_off_s"].sum()
d_f  = fdf["duel_def_f"].sum() + fdf["duel_off_f"].sum()
dds  = fdf["duel_def_s"].sum(); dos = fdf["duel_off_s"].sum()
inter= fdf["interceptions"].sum()
c_s  = fdf["cross_s"].sum(); c_f = fdf["cross_f"].sum()

pass_acc = round(p_s / (p_s + p_f) * 100, 1) if (p_s + p_f) else 0
duel_acc = round(d_s / (d_s + d_f) * 100, 1) if (d_s + d_f) else 0
cross_acc = round(c_s / (c_s + c_f) * 100, 1) if (c_s + c_f) else 0

# ══════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════
st.markdown(f"<div style='display:flex;align-items:baseline;gap:16px;'><span style='font-size:26px;font-weight:700;color:#e8ecf4;'>Performance Dashboard</span><span style='font-size:14px;color:#5a6478;'>minutos: {total_min}</span></div>", unsafe_allow_html=True)

# ── Section: Passes ──
st.markdown("<div class='section-title'>📮 Passes</div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: card_blue("📋 VISÃO GERAL", "Passes p90", fmt_p90(p_s + p_f), f"Total: <strong>{p_s + p_f}</strong>", f"Completos: <strong>{p_s}</strong>")
with c2: card_blue("📊 PRECISÃO", "% Acerto", f"{pass_acc}%", f"Certos: <strong>{p_s}</strong>", f"Errados: <strong>{p_f}</strong>")
with c3: card_blue("🎯 IMPACTO", "Passes Certos p90", fmt_p90(p_s), "", f"min: <strong>{total_min}</strong>")

# ── Section: Ações Defensivas ──
st.markdown("<div class='section-title'>🛡️ Ações Defensivas</div>", unsafe_allow_html=True)
d1, d2, d3 = st.columns(3)
acoes_def_total = inter + d_s + d_f
with d1: card_green("📋 GERAL", "Ações Def. p90", fmt_p90(acoes_def_total), f"Sucesso: <strong>{d_s}</strong>", f"Falha: <strong>{d_f}</strong>")
with d2: card_green("⚔️ DUELOS", "% Ganhos", f"{duel_acc}%", f"Def: <strong>{dds}</strong>", f"Of: <strong>{dos}</strong>")
with d3: card_green("✖️ INTERCEPTAÇÕES", "Total", str(inter), f"p90: <strong>{fmt_p90(inter)}</strong>", "")

# ── Section: Cruzamentos ──
st.markdown("<div class='section-title'>🎯 Cruzamentos</div>", unsafe_allow_html=True)
x1, x2, x3 = st.columns(3)
with x1: card_gold("📋 GERAL", "Total", str(c_s + c_f), f"p90: <strong>{fmt_p90(c_s + c_f)}</strong>", "")
with x2: card_gold("✅ SUCESSO", "Completados", str(c_s), f"<strong>{cross_acc}%</strong> acerto", "")
with x3: card_gold("❌ FALHA", "Errados", str(c_f), "", "")

# ── Data Table ──
st.markdown("---")
st.markdown("<div style='font-size:15px;font-weight:600;color:#c8ccd6;margin-bottom:12px;'>📋 Detalhamento por Partida</div>", unsafe_allow_html=True)

display_df = fdf.rename(columns={
    "match": "Partida", "min": "Min", "passes_success": "Passes OK", "passes_fail": "Passes Erro",
    "interceptions": "Intercept.", "duel_def_s": "Duelo Def OK", "duel_def_f": "Duelo Def Erro",
    "duel_off_s": "Duelo Of OK", "duel_off_f": "Duelo Of Erro",
    "cross_s": "Cruz. OK", "cross_f": "Cruz. Erro"
}).set_index("Partida")

st.dataframe(display_df, use_container_width=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Exportar CSV", csv, "performance.csv", "text/csv", use_container_width=True)
