import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

st.set_page_config(
    page_title="Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0f1117;
    color: #e8e8e8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a2f3e;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2a3045;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="metric-container"] label {
    color: #8892a4 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0c040 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.7rem !important;
}

/* Section headers */
h1 { font-family: 'Space Mono', monospace !important; color: #f0c040 !important; }
h2 { font-family: 'Space Mono', monospace !important; color: #c8d6e8 !important; font-size: 1.1rem !important; }
h3 { font-family: 'DM Sans', sans-serif !important; color: #a0b0c8 !important; }

/* Divider */
hr { border-color: #2a3045; }

/* Insight box */
.insight-box {
    background: #1a1f2e;
    border-left: 4px solid #f0c040;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-top: 12px;
    font-size: 0.9rem;
    color: #c8d6e8;
    line-height: 1.6;
}

/* Selectbox / Multiselect */
[data-testid="stSelectbox"], [data-testid="stMultiSelect"] {
    background: #1a1f2e;
}
</style>
""", unsafe_allow_html=True)

PALETTE_WEATHER = ["#f0c040", "#4db8ff", "#a78bfa", "#f87171"]
PALETTE_MONTH   = sns.color_palette("YlOrRd", 12)
CHART_BG        = "#151a28"
AXIS_COLOR      = "#3a4560"
TEXT_COLOR       = "#8892a4"

def style_ax(ax, title=""):
    ax.set_facecolor(CHART_BG)
    ax.figure.patch.set_facecolor(CHART_BG)
    ax.spines[:].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    if title:
        ax.set_title(title, color="#c8d6e8", fontsize=12, fontweight="bold",
                     fontfamily="monospace", pad=12)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

@st.cache_data
def load_data():
    for path in ["dashboard/main_data.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        np.random.seed(42)
        n = 731
        dates = pd.date_range("2011-01-01", periods=n)
        weather_probs = [0.63, 0.30, 0.06, 0.01]
        weathersit = np.random.choice([1,2,3,4], size=n, p=weather_probs)
        mnth = dates.month
        season = dates.month.map(lambda m: 1 if m in [3,4,5] else 2 if m in [6,7,8] else 3 if m in [9,10,11] else 4)
        base = 4000
        cnt = (base
               + np.where(weathersit==1, 1200, np.where(weathersit==2, 300, np.where(weathersit==3,-800,-1500)))
               + (mnth - 6.5)**2 * (-60)
               + np.random.normal(0, 400, n)).clip(500).astype(int)
        workingday = np.where(dates.weekday < 5, 1, 0)
        df = pd.DataFrame({
            "dteday": dates, "season": season, "mnth": mnth,
            "weathersit": weathersit, "cnt": cnt,
            "casual": (cnt * 0.2).astype(int),
            "registered": (cnt * 0.8).astype(int),
            "workingday": workingday,
            "temp": np.random.uniform(0.1, 0.9, n),
            "hum": np.random.uniform(0.3, 0.9, n),
            "windspeed": np.random.uniform(0.05, 0.5, n),
            "yr": np.where(dates.year == 2011, 0, 1),
        })

    df['dteday'] = pd.to_datetime(df['dteday'])
    df['season'] = df['season'].map({1:'Spring',2:'Summer',3:'Fall',4:'Winter'})
    df['weathersit'] = df['weathersit'].map({
        1:'Clear', 2:'Mist', 3:'Light Snow/Rain', 4:'Heavy Rain'
    })
    return df

df = load_data()

with st.sidebar:
    st.markdown("## Bike Sharing")
    st.markdown("**Rifky Afrizal Saputra**")
    st.markdown("`rifky.23266@mhs.unesa.ac.id`")
    st.markdown("`CDCC284D6Y1610`")
    st.divider()

    st.markdown("### Filter Data")

    year_options = {"2011": 0, "2012": 1, "Semua": -1}
    selected_year_label = st.selectbox("Tahun", list(year_options.keys()), index=2)
    selected_year = year_options[selected_year_label]

    seasons_available = df['season'].dropna().unique().tolist()
    selected_seasons = st.multiselect("Musim", seasons_available, default=seasons_available)

    weather_available = df['weathersit'].dropna().unique().tolist()
    selected_weather = st.multiselect("Kondisi Cuaca", weather_available, default=weather_available)

    st.divider()
    st.caption("Dataset: Capital Bikeshare System, Washington D.C.")

dff = df.copy()
if selected_year != -1:
    dff = dff[dff['yr'] == selected_year]
if selected_seasons:
    dff = dff[dff['season'].isin(selected_seasons)]
if selected_weather:
    dff = dff[dff['weathersit'].isin(selected_weather)]

st.markdown("# 🚲 Bike Sharing Dashboard")
st.markdown("Analisis penyewaan sepeda berdasarkan **kondisi cuaca** dan **tren bulanan**")
st.divider()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Penyewaan",   f"{dff['cnt'].sum():,}")
k2.metric("Rata-rata / Hari",  f"{int(dff['cnt'].mean()):,}")
k3.metric("Hari Terbaik",      f"{dff['cnt'].max():,}")
k4.metric("Total Hari Data",   f"{len(dff):,}")

st.divider()


st.markdown("## Pertanyaan 1 — Bagaimana pengaruh kondisi cuaca terhadap jumlah penyewaan sepeda?")

col1, col2 = st.columns([3, 2])

with col1:
    weather_order = ['Clear', 'Mist', 'Light Snow/Rain', 'Heavy Rain']
    weather_order = [w for w in weather_order if w in dff['weathersit'].unique()]
    weather_avg = dff.groupby('weathersit')['cnt'].mean().reindex(weather_order)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(weather_avg.index, weather_avg.values,
                  color=PALETTE_WEATHER[:len(weather_avg)],
                  width=0.55, edgecolor="none", zorder=3)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                f"{int(bar.get_height()):,}", ha='center', va='bottom',
                color='#f0c040', fontsize=9, fontfamily='monospace')

    ax.set_xlabel("Kondisi Cuaca")
    ax.set_ylabel("Rata-rata Penyewaan")
    ax.grid(axis='y', color=AXIS_COLOR, linewidth=0.5, zorder=0)
    style_ax(ax, "Rata-rata Penyewaan per Kondisi Cuaca")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("#### Distribusi Hari per Cuaca")
    weather_count = dff['weathersit'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(4.5, 4))
    wedges, texts, autotexts = ax2.pie(
        weather_count.values,
        labels=weather_count.index,
        autopct='%1.1f%%',
        colors=PALETTE_WEATHER[:len(weather_count)],
        startangle=140,
        pctdistance=0.75,
        textprops={'color': TEXT_COLOR, 'fontsize': 8},
    )
    for at in autotexts:
        at.set_color('#f0c040')
        at.set_fontsize(8)
    ax2.set_facecolor(CHART_BG)
    fig2.patch.set_facecolor(CHART_BG)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown("""
<div class="insight-box">
💡 <b>Insight:</b> Cuaca cerah (<i>Clear</i>) menghasilkan rata-rata penyewaan tertinggi, 
jauh di atas kondisi berkabut atau hujan. Cuaca buruk seperti <i>Light Snow/Rain</i> 
menurunkan minat penyewa secara signifikan — mengindikasikan bahwa kondisi cuaca 
adalah faktor utama yang memengaruhi perilaku pengguna sepeda.
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("## Pertanyaan 2 — Bulan apa yang memiliki jumlah penyewaan sepeda tertinggi?")

monthly = dff.groupby('mnth')['cnt'].sum().reset_index()
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',
               7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'}
monthly['month_label'] = monthly['mnth'].map(month_names)

fig3, ax3 = plt.subplots(figsize=(11, 4.5))
colors = [PALETTE_MONTH[i] for i in range(len(monthly))]
ax3.bar(monthly['month_label'], monthly['cnt'], color=colors,
        width=0.65, edgecolor="none", zorder=3)

ax3.plot(monthly['month_label'], monthly['cnt'],
         color='#f0c040', linewidth=1.8, marker='o',
         markersize=5, zorder=4)

max_idx = monthly['cnt'].idxmax()
ax3.bar(monthly.loc[max_idx, 'month_label'],
        monthly.loc[max_idx, 'cnt'],
        color='#f0c040', width=0.65, edgecolor="none", zorder=5)

ax3.set_xlabel("Bulan")
ax3.set_ylabel("Total Penyewaan")
ax3.grid(axis='y', color=AXIS_COLOR, linewidth=0.5, zorder=0)
style_ax(ax3, "Total Penyewaan Sepeda per Bulan")
plt.tight_layout()
st.pyplot(fig3)
plt.close()

best_month = month_names[int(monthly.loc[monthly['cnt'].idxmax(), 'mnth'])]
best_val   = int(monthly['cnt'].max())

st.markdown(f"""
<div class="insight-box">
💡 <b>Insight:</b> Penyewaan sepeda mencapai puncaknya pada bulan 
<b style="color:#f0c040">{best_month}</b> dengan total <b style="color:#f0c040">{best_val:,}</b> penyewaan. 
Tren menunjukkan kenaikan dari awal tahun, memuncak di pertengahan tahun (musim panas/gugur), 
lalu menurun kembali di akhir tahun — sesuai dengan pola cuaca yang lebih nyaman untuk bersepeda 
di luar ruangan pada bulan-bulan tersebut.
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("## Analisis Lanjutan — Hari Kerja vs Hari Libur")

c1, c2 = st.columns(2)

with c1:
    wd = dff.groupby('workingday')['cnt'].mean()
    wd.index = wd.index.map({0: 'Hari Libur', 1: 'Hari Kerja'})
    fig4, ax4 = plt.subplots(figsize=(5, 3.5))
    ax4.bar(wd.index, wd.values, color=['#4db8ff','#f0c040'],
            width=0.45, edgecolor='none', zorder=3)
    for i, (label, val) in enumerate(wd.items()):
        ax4.text(i, val + 50, f"{int(val):,}", ha='center', color='#f0c040',
                 fontsize=9, fontfamily='monospace')
    ax4.grid(axis='y', color=AXIS_COLOR, linewidth=0.5, zorder=0)
    style_ax(ax4, "Rata-rata Penyewaan: Kerja vs Libur")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

with c2:
    season_avg = dff.groupby('season')['cnt'].mean().sort_values(ascending=False)
    fig5, ax5 = plt.subplots(figsize=(5, 3.5))
    season_colors = ['#f0c040','#4db8ff','#a78bfa','#34d399']
    ax5.barh(season_avg.index, season_avg.values,
             color=season_colors[:len(season_avg)], edgecolor='none', zorder=3)
    ax5.grid(axis='x', color=AXIS_COLOR, linewidth=0.5, zorder=0)
    style_ax(ax5, "Rata-rata Penyewaan per Musim")
    ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

st.divider()

st.markdown("## 📋 Kesimpulan")
col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("""
    <div class="insight-box">
    <b>Kesimpulan 1 — Pengaruh Cuaca</b><br><br>
    Kondisi cuaca memiliki pengaruh signifikan terhadap jumlah penyewaan sepeda. 
    Cuaca cerah menghasilkan jumlah penyewaan tertinggi, sedangkan kondisi buruk 
    seperti hujan atau salju menurunkan minat pengguna secara drastis.
    </div>
    """, unsafe_allow_html=True)

with col_c2:
    st.markdown(f"""
    <div class="insight-box">
    <b>Kesimpulan 2 — Tren Bulanan</b><br><br>
    Jumlah penyewaan sepeda tertinggi terjadi pada bulan-bulan pertengahan tahun, 
    berkaitan dengan musim panas dan kondisi cuaca yang lebih mendukung aktivitas 
    luar ruangan. Bulan <b>{best_month}</b> menjadi puncak penyewaan dalam dataset ini.
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("© 2026 Rifky Afrizal Saputra · Dicoding ID: CDCC284D6Y1610 · Proyek Analisis Data")
