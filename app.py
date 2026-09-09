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
RECENT_YEARS = 20       # 최근 비교 기간


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
    yearly = df.groupby("연도").agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    ).reset_index()

    yearly = yearly[
        (yearly["연도"] <= BASE_YEAR_LIMIT) &
        (yearly["관측일수"] >= MIN_OBS_DAYS)
    ].reset_index(drop=True)

    return yearly


def run_regression(yearly_subset):
    """연도별 자료로 회귀분석을 수행하고 결과를 딕셔너리로 반환"""
    x = yearly_subset["연도"].values
    y = yearly_subset["평균기온"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "r_squared": r_value ** 2,
        "n_years": len(yearly_subset),
        "start_year": int(yearly_subset["연도"].min()),
        "end_year": int(yearly_subset["연도"].max()),
        "slope_per_100yr": slope * 100
    }


df_raw = load_data()
yearly = get_yearly_avg(df_raw)

if yearly.empty:
    st.error("조건을 만족하는 데이터가 없습니다.")
    st.stop()

# ------------------------------------------------------
# 전체 기간 회귀분석
# ------------------------------------------------------
full_result = run_regression(yearly)

# ------------------------------------------------------
# 최근 20년 회귀분석
# ------------------------------------------------------
recent_cutoff = full_result["end_year"] - RECENT_YEARS + 1
recent_yearly = yearly[yearly["연도"] >= recent_cutoff].reset_index(drop=True)

if len(recent_yearly) >= 2:
    recent_result = run_regression(recent_yearly)
else:
    recent_result = None

# ------------------------------------------------------
# 화면 표시: 기본 정보
# ------------------------------------------------------
st.subheader("📊 회귀 분석에 사용된 자료")

col1, col2, col3 = st.columns(3)
col1.metric("사용된 연도 수", f"{full_result['n_years']}개")
col2.metric("시작 연도", f"{full_result['start_year']}년")
col3.metric("끝 연도", f"{full_result['end_year']}년")

# ------------------------------------------------------
# 100년당 기온 상승폭 비교 (전체 vs 최근 20년)
# ------------------------------------------------------
st.subheader("🌡️ 100년당 기온 상승폭 비교")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 전체 기간")
    st.markdown(
        f"""
        <div style="text-align:center;">
            <span style="font-size:16px; color:gray;">
                {full_result['start_year']}년 ~ {full_result['end_year']}년
                ({full_result['n_years']}개 연도)
            </span><br>
            <span style="font-size:48px; font-weight:bold; color:#1f77b4;">
                {full_result['slope_per_100yr']:+.2f}°C
            </span><br>
            <span style="font-size:16px;">/ 100년</span><br>
            <span style="font-size:14px; color:gray;">
                상관계수 r = {full_result['r_value']:.3f}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_b:
    st.markdown(f"#### 최근 {RECENT_YEARS}년")
    if recent_result is not None:
        st.markdown(
            f"""
            <div style="text-align:center;">
                <span style="font-size:16px; color:gray;">
                    {recent_result['start_year']}년 ~ {recent_result['end_year']}년
                    ({recent_result['n_years']}개 연도)
                </span><br>
                <span style="font-size:48px; font-weight:bold; color:#d62728;">
                    {recent_result['slope_per_100yr']:+.2f}°C
                </span><br>
                <span style="font-size:16px;">/ 100년</span><br>
                <span style="font-size:14px; color:gray;">
                    상관계수 r = {recent_result['r_value']:.3f}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("최근 자료가 충분하지 않습니다.")

st.caption(
    "※ 최근 기간의 기울기가 전체 기간보다 크다면, 최근 들어 기온 상승 속도가 "
    "빨라지고 있다는 뜻으로 해석할 수 있어요."
)

# ------------------------------------------------------
# 산점도 + 회귀 직선 (전체 vs 최근, Plotly)
# ------------------------------------------------------
st.subheader("📈 연도별 평균기온 산점도와 회귀 직선")

fig = go.Figure()

# 전체 산점도
fig.add_trace(go.Scatter(
    x=yearly["연도"], y=yearly["평균기온"],
    mode="markers",
    name="연평균기온(전체)",
    marker=dict(color="royalblue", size=7)
))

# 최근 20년 강조 산점도
if recent_result is not None:
    fig.add_trace(go.Scatter(
        x=recent_yearly["연도"], y=recent_yearly["평균기온"],
        mode="markers",
        name=f"최근 {RECENT_YEARS}년 자료",
        marker=dict(color="orange", size=9, symbol="diamond")
    ))

# 회귀 직선 (전체)
x_line = np.linspace(1900, 2100, 500)
y_line_full = full_result["slope"] * x_line + full_result["intercept"]

fig.add_trace(go.Scatter(
    x=x_line, y=y_line_full,
    mode="lines",
    name="회귀 직선 (전체 기간)",
    line=dict(color="#1f77b4", dash="dash")
))

# 회귀 직선 (최근 20년)
if recent_result is not None:
    y_line_recent = recent_result["slope"] * x_line + recent_result["intercept"]
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line_recent,
        mode="lines",
        name=f"회귀 직선 (최근 {RECENT_YEARS}년)",
        line=dict(color="#d62728", dash="dot")
    ))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------
# 슬라이더로 연도 선택 -> 예측 기온 표시 (전체 기간 회귀식 기준)
# ------------------------------------------------------
st.subheader("🔮 연도별 예상 기온 확인 (전체 기간 회귀식 기준)")

selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = full_result["slope"] * selected_year + full_result["intercept"]

st.markdown(
    f"""
    <div style="text-align:center; margin-top:20px;">
        <span style="font-size:22px;">{selected_year}년 예상 평균기온</span><br>
        <span style="font-size:60px; font-weight:bold; color:#2ca02c;">
            {predicted_temp:.2f} °C
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    f"※ 이 예측은 {full_result['start_year']}년~{full_result['end_year']}년까지 "
    f"{full_result['n_years']}개 연도의 자료를 바탕으로 한 "
    f"단순 선형회귀 모델(전체 기간)의 결과이며, 실제와 다를 수 있습니다."
)
