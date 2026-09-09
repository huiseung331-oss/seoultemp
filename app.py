import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# ------------------------------------------------------
# 기본 설정
# ------------------------------------------------------
st.set_page_config(page_title="기온 예측기", layout="centered")
st.title("🌡️ 서울 연평균 기온 예측기")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
BASE_YEAR_LIMIT = 2025  # 수업 기준 기간
MIN_OBS_DAYS = 300      # 연간 최소 관측일수


# ------------------------------------------------------
# 데이터 불러오기 및 전처리
# ------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    return df


@st.cache_data
def get_yearly_avg(df):
    # 연도별 평균기온과 관측일수 계산
    yearly = df.groupby("연도").agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    ).reset_index()

    # 조건: 2025년까지 + 관측일수 300일 이상
    yearly = yearly[
        (yearly["연도"] <= BASE_YEAR_LIMIT) &
        (yearly["관측일수"] >= MIN_OBS_DAYS)
    ].reset_index(drop=True)

    return yearly


df_raw = load_data()
yearly = get_yearly_avg(df_raw)

if yearly.empty:
    st.error("조건을 만족하는 데이터가 없습니다.")
    st.stop()

# ------------------------------------------------------
# 회귀 분석 (연도 -> 평균기온)
# ------------------------------------------------------
x = yearly["연도"].values
y = yearly["평균기온"].values

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# ------------------------------------------------------
# 화면 표시: 회귀 정보
# ------------------------------------------------------
st.subheader("📊 회귀 분석 정보")

col1, col2, col3 = st.columns(3)
col1.metric("사용된 연도 수", f"{n_years}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")

st.write(f"**상관계수 (r)**: {r_value:.4f}  |  **결정계수 (R²)**: {r_squared:.4f}")
st.write(f"**회귀식**: 평균기온 = {slope:.5f} × 연도 + {intercept:.3f}")

# ------------------------------------------------------
# 산점도 + 회귀 직선 (Plotly)
# ------------------------------------------------------
st.subheader("📈 연도별 평균기온 산점도와 회귀 직선")

fig = go.Figure()

# 산점도
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode="markers",
    name="실제 연평균기온",
    marker=dict(color="royalblue", size=7)
))

# 회귀 직선 (실측 구간 + 예측을 위해 넉넉하게 확장)
x_line = np.linspace(1900, 2100, 500)
y_line = slope * x_line + intercept

fig.add_trace(go.Scatter(
    x=x_line, y=y_line,
    mode="lines",
    name="회귀 직선",
    line=dict(color="firebrick", dash="dash")
))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------
# 슬라이더로 연도 선택 -> 예측 기온 표시
# ------------------------------------------------------
st.subheader("🔮 연도별 예상 기온 확인")

selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = slope * selected_year + intercept

st.markdown(
    f"""
    <div style="text-align:center; margin-top:20px;">
        <span style="font-size:22px;">{selected_year}년 예상 평균기온</span><br>
        <span style="font-size:60px; font-weight:bold; color:#d62728;">
            {predicted_temp:.2f} °C
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    f"※ 이 예측은 {start_year}년~{end_year}년까지 {n_years}개 연도의 자료를 바탕으로 한 "
    f"단순 선형회귀 모델의 결과이며, 실제와 다를 수 있습니다."
)
