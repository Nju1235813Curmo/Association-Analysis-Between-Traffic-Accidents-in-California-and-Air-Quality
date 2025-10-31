import os
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ------------------- 1. 自动读取原始数据 -------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
accident_path = os.path.join(current_dir, "US_Accidents_March23.csv")
aqi_path = os.path.join(current_dir, "US_AQI.csv")

print("正在加载数据中，请稍候...")

acc = pd.read_csv(accident_path, usecols=["Start_Time", "City", "State", "Severity", "Start_Lat", "Start_Lng"], low_memory=False)
acc = acc.dropna(subset=["Start_Time", "City", "Start_Lat", "Start_Lng"])
acc["Start_Time"] = acc["Start_Time"].astype(str).str.replace(r"\.000000000", "", regex=True)
acc["Date"] = pd.to_datetime(acc["Start_Time"], errors="coerce").dt.date
acc["Year"] = pd.to_datetime(acc["Start_Time"], errors="coerce").dt.year
print(f"✅ 加州事故数据加载完成，共 {len(acc)} 条记录")

aqi = pd.read_csv(aqi_path, low_memory=False)
aqi.rename(columns={"city_ascii": "City", "state_id": "State"}, inplace=True)
aqi["Date"] = pd.to_datetime(aqi["Date"], errors="coerce").dt.date

# ------------------- 2. 聚合并合并 -------------------
acc_daily = acc.groupby(["Date", "City"]).agg({
    "Severity": "mean",
    "Start_Lat": "mean",
    "Start_Lng": "mean",
    "Start_Time": "count"
}).reset_index().rename(columns={"Start_Time": "Accident_Count"})

aqi_daily = aqi.groupby(["Date", "City"]).agg({"AQI": "mean"}).reset_index()

merged = pd.merge(acc_daily, aqi_daily, on=["Date", "City"], how="inner")
print(f"合并后数据量：{len(merged)} 行")

merged["Year"] = pd.to_datetime(merged["Date"]).dt.year

# ------------------- 3. 构建 Dash App -------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "加州交通事故与空气质量交互仪表盘"

city_options = [{"label": c, "value": c} for c in sorted(merged["City"].unique())]
year_options = [{"label": str(y), "value": y} for y in sorted(merged["Year"].unique())]

# ------------------- 4. 布局 -------------------
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("🚗 加州交通事故与空气质量交互仪表盘", className="text-center"))
    ], className="mt-3 mb-2"),

    dbc.Row([
        dbc.Col([
            html.Label("选择城市："),
            dcc.Dropdown(id="city-dropdown", options=city_options, value=None, clearable=True)
        ], width=6),
        dbc.Col([
            html.Label("选择年份："),
            dcc.Dropdown(id="year-dropdown", options=year_options, value=2022, clearable=False)
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Col([html.H4("📍 事故地理分布（Folium地图）"), html.Div(id="map-container")], width=6),
            html.Iframe(id="map", width="100%", height="500")
        ], width=6),
        dbc.Col([
            html.H4("📈 事故与空气质量动态变化（Plotly动画图）"),
            dcc.Graph(id="animation-graph", style={"height": "500px"})
        ], width=6)
    ]),

    html.Div(id='summary-text', style={
        'padding': '20px',
        'backgroundColor': '#f9f9f9',
        'borderTop': '2px solid #ccc',
        'fontSize': '16px',
        'lineHeight': '1.8',
    })
], fluid=True)


# ------------------- 5. 回调函数 -------------------
@app.callback(
    [Output("map", "srcDoc"),
     Output("animation-graph", "figure")],
    [Input("city-dropdown", "value"),
     Input("year-dropdown", "value")]
)
def update_dashboard(selected_city, selected_year):
    df = merged[merged["Year"] == selected_year]
    if selected_city:
        df = df[df["City"] == selected_city]

    # ========== 地图 ==========
    if df.empty:
        return "<h4 style='color:red;'>暂无数据</h4>", px.scatter()

    center = [df["Start_Lat"].mean(), df["Start_Lng"].mean()]
    fmap = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(fmap)
    for _, row in df.iterrows():
        popup = f"城市：{row['City']}<br>日期：{row['Date']}<br>事故数：{row['Accident_Count']}<br>AQI：{row['AQI']:.1f}"
        folium.CircleMarker(
            location=[row["Start_Lat"], row["Start_Lng"]],
            radius=max(3, min(row["Accident_Count"]/10, 10)),
            color="red" if row["Severity"] > 2.5 else "blue",
            fill=True, fill_opacity=0.6, popup=popup
        ).add_to(cluster)
    fmap_html = fmap.get_root().render()

    # ========== 动画图 ==========
    # 每月聚合（避免每日帧太多卡顿）
    df["Month"] = pd.to_datetime(df["Date"]).dt.to_period("M").astype(str)
    df_grouped = df.groupby(["Month", "City"]).agg({
        "Accident_Count": "sum",
        "AQI": "mean"
    }).reset_index()

    fig = px.scatter(
        df_grouped,
        x="AQI", y="Accident_Count",
        animation_frame="Month",
        color="City",
        size="Accident_Count",
        hover_name="City",
        title=f"{selected_year}年加州城市交通事故与空气质量动态关系", 
        labels={"AQI": "空气质量指数", "Accident_Count": "事故数量"},
        template="plotly_white"
    )

    fig.update_layout(
        transition_duration=500,
        xaxis=dict(title="空气质量指数 (AQI)", range=[0, df_grouped["AQI"].max() * 1.2]),
        yaxis=dict(title="事故数量", range=[0, df_grouped["Accident_Count"].max() * 1.2])
    )

    return fmap_html, fig

    # ========== 文本分析与摘要 ==========
@app.callback(
    Output('summary-text', 'children'),
    Input('year-dropdown', 'value'),
    Input('city-dropdown', 'value')
)
def update_summary(selected_year, selected_city):
    filtered = merged[merged["Year"] == selected_year]
    if selected_city:
        filtered = filtered[filtered["City"] == selected_city]

    if filtered.empty:
        return html.P("⚠️ 当前条件下无数据，无法生成分析摘要", style={'color': 'red'})

    avg_aqi = filtered['AQI'].mean()
    total_accidents = filtered['Accident_Count'].sum()
    avg_severity = filtered['Severity'].mean()
    if filtered['AQI'].notna().sum() > 0 and filtered['Severity'].notna().sum() > 0:
        corr = filtered['AQI'].corr(filtered['Severity'])
    else:
        corr = 0

    if avg_aqi < 50:
        aqi_level = "良"
    elif avg_aqi < 100:
        aqi_level = "轻度污染"
    else:
        aqi_level = "中度污染"
        
    city_text = f" {selected_city}市" if selected_city else "全加州"
    summary = [
        html.H4("📊 分析摘要", style={'marginBottom': '10px'}),
        html.P(f"{selected_year}年{city_text}平均空气质量指数（AQI）为 {avg_aqi:.1f}（{aqi_level}），"
               f"共发生 {total_accidents:,} 起交通事故。"),
        html.P(f"事故平均严重程度为 {avg_severity:.2f}。"
               f"AQI 与事故严重程度之间的相关系数为 {corr:.2f}，"
               f"表明{'弱正' if 0<corr<0.3 else '较强正' if corr>=0.3 else '弱负' if -0.3<corr<0 else '较强负'}相关关系。"),
        html.P("整体来看，空气质量对交通安全存在一定影响，"
               "但仍需结合季节与城市交通特征进一步分析。")
    ]
    return summary


# ------------------- 6. 启动 Dash -------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050, use_reloader=False)
