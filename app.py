import streamlit as st
import pandas as pd

st.set_page_config(page_title="Scout Dashboard", layout="wide")

# Dados dos jogos
data = {
    "vs Salt Lake": {"min": 85, "p_suc": 16, "p_fal": 4, "int": 3, "d_def_s": 4, "d_def_f": 3, "d_of_s": 0, "d_of_f": 1, "c_s": 1, "c_f": 2},
    "vs Vancouver": {"min": 45, "p_suc": 19, "p_fal": 3, "int": 3, "d_def_s": 1, "d_def_f": 0, "d_of_s": 0, "d_of_f": 0, "c_s": 0, "c_f": 0},
    "vs LAFC":       {"min": 82, "p_suc": 13, "p_fal": 1, "int": 3, "d_def_s": 3, "d_def_f": 4, "d_of_s": 2, "d_of_f": 0, "c_s": 0, "c_f": 1}
}

def get_stats(df):
    total_min = df['min'].sum()
    p_s = df['p_suc'].sum(); p_f = df['p_fal'].sum()
    d_def_s = df['d_def_s'].sum(); d_def_f = df['d_def_f'].sum()
    d_of_s = df['d_of_s'].sum(); d_of_f = df['d_of_f'].sum()
    d_s = d_def_s + d_of_s; d_f = d_def_f + d_of_f
    return {
        "min": total_min,
        "p_p90": round((p_s + p_f) / total_min * 90, 1),
        "p_acc": round(p_s / (p_s + p_f) * 100, 1) if (p_s + p_f) > 0 else 0,
        "p_total": p_s + p_f, "p_suc": p_s,
        "d_p90": round((d_s + d_f) / total_min * 90, 1),
        "d_acc": round(d_s / (d_s + d_f) * 100, 1) if (d_s + d_f) > 0 else 0,
        "d_def_total": d_def_s + d_def_f, "d_def_s": d_def_s, "d_def_f": d_def_f,
        "d_of_total": d_of_s + d_of_f, "d_of_s": d_of_s, "d_of_f": d_of_f,
        "int": int(df['int'].sum()),
        "c_s": int(df['c_s'].sum()), "c_f": int(df['c_f'].sum()),
        "c_total": int(df['c_s'].sum() + df['c_f'].sum())
    }

# UI
st.title("📊 Scout Performance Dashboard")
st.markdown("Análise de performance por partida — métricas p90 e taxas de acerto")

match = st.selectbox("Selecione a Partida", ["Todos os jogos"] + list(data.keys()))
df = pd.DataFrame(data).T if match == "Todos os jogos" else pd.DataFrame([data[match]], index=[match])
stats = get_stats(df)

# CSS customizado
st.markdown("""
<style>
    .card { padding: 22px 20px; border-radius: 12px; color: white; text-align: center; margin: 8px 0; height: 160px; display: flex; flex-direction: column; justify-content: center; }
    .card h4 { margin: 0 0 6px 0; font-size: 13px; text-transform: uppercase; letter-spacing: 1.2px; opacity: 0.85; }
    .card .label { font-size: 14px; color: #B0B0B0; margin: 4px 0; }
    .card .value { font-size: 32px; font-weight: 700; margin: 4px 0; }
    .card .sub { font-size: 13px; opacity: 0.7; margin: 2px 0; }
    .total-bar { display: flex; justify-content: space-between; font-size: 13px; padding: 0 8px; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

# Linha 1 — Passes
st.subheader("📮 Passes")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='card' style='background:#1E3A5F'><h4>📋 Visão Geral</h4><div class='label'>Passes p90</div><div class='value'>{stats['p_p90']}</div><div class='total-bar'><span>Total: {stats['p_total']}</span><span>Completos: {stats['p_suc']}</span></div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='card' style='background:#2D5016'><h4>📊 Precisão</h4><div class='label'>% Acerto</div><div class='value'>{stats['p_acc']}%</div><div class='total-bar'><span>{stats['p_suc']} certos</span><span>{stats['p_total'] - stats['p_suc']} errados</span></div></div>", unsafe_allow_html=True)
with col3:
    p90_completados = round(stats['p_suc'] / stats['min'] * 90, 1) if stats['min'] > 0 else 0
    st.markdown(f"<div class='card' style='background:#8B6914'><h4>🎯 Impacto</h4><div class='label'>Passes Certos p90</div><div class='value'>{p90_completados}</div><div class='sub'>minutos: {stats['min']}</div></div>", unsafe_allow_html=True)

# Linha 2 — Ações Defensivas
st.subheader("🛡️ Ações Defensivas")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='card' style='background:#1E3A5F'><h4>📋 Geral</h4><div class='label'>Ações Defensivas p90</div><div class='value'>{stats['d_p90']}</div><div class='total-bar'><span>Sucesso: {stats['d_def_s'] + stats['d_of_s']}</span><span>Falha: {stats['d_def_f'] + stats['d_of_f']}</span></div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='card' style='background:#2D5016'><h4>⚔️ Duelos</h4><div class='label'>% Duelos Ganhos</div><div class='value'>{stats['d_acc']}%</div><div class='total-bar'><span>Def: {stats['d_def_s']}/{stats['d_def_total']}</span><span>Of: {stats['d_of_s']}/{stats['d_of_total']}</span></div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='card' style='background:#8B6914'><h4>✖️ Interceptações</h4><div class='label'>Total</div><div class='value'>{stats['int']}</div><div class='sub'>p90: {round(stats['int'] / stats['min'] * 90, 1)}</div></div>", unsafe_allow_html=True)

# Linha 3 — Cruzamentos
st.subheader("🎯 Cruzamentos")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='card' style='background:#1E3A5F'><h4>📋 Geral</h4><div class='label'>Total Cruzamentos</div><div class='value'>{stats['c_total']}</div><div class='total-bar'><span>p90: {round(stats['c_total'] / stats['min'] * 90, 1)}</span></div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='card' style='background:#2D5016'><h4>✅ Sucesso</h4><div class='label'>Completados</div><div class='value'>{stats['c_s']}</div><div class='sub'>{round(stats['c_s'] / stats['c_total'] * 100, 0) if stats['c_total'] > 0 else 0}% de acerto</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='card' style='background:#8B6914'><h4>❌ Falha</h4><div class='label'>Errados</div><div class='value'>{stats['c_f']}</div></div>", unsafe_allow_html=True)

st.divider()

# Tabela detalhada
st.subheader("📋 Detalhamento por Partida")
df_detail = df.copy()
df_detail.columns = ["Min", "Passes OK", "Passes Erro", "Intercept.", "Duelo Def OK", "Duelo Def Erro", "Duelo Of OK", "Duelo Of Erro", "Cruz. OK", "Cruz. Erro"]
df_detail.index.name = "Partida"
st.dataframe(df_detail, use_container_width=True)

# Botão exportar
csv = df.to_csv(index=True).encode('utf-8')
st.download_button("📥 Exportar CSV", csv, "performance.csv", "text/csv", use_container_width=True)
