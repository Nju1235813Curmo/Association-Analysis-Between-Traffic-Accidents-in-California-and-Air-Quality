"""
加州地区空气质量与交通事故关联分析
使用Kaggle的US Air Quality和US Accidents数据集
研究空气质量（特别是能见度）对交通事故的影响
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.patches import Patch
import folium
from folium.plugins import HeatMap
import os
import gc

warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("加州地区空气质量与交通事故关联分析")
print("数据来源：")
print("1. US Accidents (Kaggle)")
print("2. US Air Quality 1980-Present (Kaggle)")
print("=" * 80)

# ==================== 1. 数据加载与预处理 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir) 
print("\n1. 正在加载数据...")

def load_accident_data():
    """
    加载US Accidents数据集
    数据来源：https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
    """
    print("   - 加载交通事故数据...")
    
    # 尝试多个可能的文件名
    possible_files = [
        os.path.join(current_dir, 'US_Accidents_March23.csv'),
        os.path.join(current_dir, 'US_Accidents.csv'),
        'US_Accidents.csv',
        'US_Accidents_2023.csv',
        r'D:/南京大学/信管期末/大三上/大数据/project/data/US_Accidents_March23.csv',
    ]
    
    accidents_file = None
    for file in possible_files:
        if os.path.exists(file):
            accidents_file = file
            print(f"   找到事故数据文件: {file}")
            break
    
    if accidents_file is None:
        print("   警告：未找到事故数据文件，将使用模拟数据")
        return None
    
    # 读取必要的列
    try:
        accidents_df = pd.read_csv(accidents_file, 
                                   usecols=['ID', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 
                                           'Start_Lng', 'City', 'County', 'State', 
                                           'Weather_Condition', 'Visibility(mi)', 
                                           'Temperature(F)', 'Humidity(%)', 
                                           'Pressure(in)', 'Wind_Speed(mph)',
                                           'Precipitation(in)', 'Weather_Timestamp'])
    except:
        # 如果某些列不存在，尝试读取基本列
        print("   注意：部分列不存在，尝试读取基本列...")
        accidents_df = pd.read_csv(accidents_file, 
                                   usecols=['ID', 'Severity', 'Start_Time', 'End_Time','Start_Lat', 
                                           'Start_Lng', 'City', 'State', 
                                           'Visibility(mi)', 'Weather_Condition'])
    
    # 筛选加州数据
    ca_accidents = accidents_df[accidents_df['State'] == 'CA'].copy()
    print(f"   - 加州交通事故记录数: {len(ca_accidents):,}")
    
    # 处理时间格式
    print("   - 处理时间格式（Start_Time + End_Time）...")
    try:
        ca_accidents['Start_Time'] = pd.to_datetime(ca_accidents['Start_Time'], format='ISO8601')
    except:
        try:
            ca_accidents['Start_Time'] = pd.to_datetime(ca_accidents['Start_Time'], format='mixed')
        except:
            ca_accidents['Start_Time'] = pd.to_datetime(ca_accidents['Start_Time'], errors='coerce')
    
    try:
        ca_accidents['End_Time'] = pd.to_datetime(ca_accidents['End_Time'], format='ISO8601')
    except:
        try:
            ca_accidents['End_Time'] = pd.to_datetime(ca_accidents['End_Time'], format='mixed')
        except:
            ca_accidents['End_Time'] = pd.to_datetime(ca_accidents['End_Time'], errors='coerce')
    
    # 移除无效的时间数据
    ca_accidents = ca_accidents.dropna(subset=['Start_Time', 'End_Time'])
    print(f"   - 清理后有效记录数: {len(ca_accidents):,}")
    
    # 添加时间特征
    ca_accidents['Year'] = ca_accidents['Start_Time'].dt.year
    ca_accidents['Month'] = ca_accidents['Start_Time'].dt.month
    ca_accidents['Hour'] = ca_accidents['Start_Time'].dt.hour
    ca_accidents['Date'] = ca_accidents['Start_Time'].dt.date
    ca_accidents['DayOfWeek'] = ca_accidents['Start_Time'].dt.dayofweek
    
    # 处理能见度数据
    if 'Visibility(mi)' in ca_accidents.columns:
        ca_accidents['Visibility(mi)'] = pd.to_numeric(ca_accidents['Visibility(mi)'], errors='coerce')
        # 移除异常值（能见度不可能超过20英里）
        ca_accidents.loc[ca_accidents['Visibility(mi)'] > 20, 'Visibility(mi)'] = np.nan
    
    return ca_accidents

def load_air_quality_data():
    """
    加载US Air Quality数据集
    数据来源：https://www.kaggle.com/datasets/calebreigada/us-air-quality-1980present
    """
    print("   - 加载空气质量数据...")
    
    # 尝试多个可能的文件名
    possible_files = [
        os.path.join(current_dir, 'US_AQI.csv'),
        os.path.join(current_dir, 'us_air_quality.csv'),
        'daily_aqi_by_county.csv',
        'annual_aqi_by_county.csv',
        r'D:/南京大学/信管期末/大三上/大数据/project/data/US_AQI.csv'
    ]
    
    air_file = None
    for file in possible_files:
        if os.path.exists(file):
            air_file = file
            print(f"   找到空气质量数据文件: {file}")
            break
    
    if air_file is None:
        print("   警告：未找到空气质量数据文件，将使用模拟数据")
        return None
    
    # 读取数据
    try:
        sample_df = pd.read_csv(air_file, nrows=5)
        print(f"   空气质量数据列: {list(sample_df.columns)}")
        
        # 根据实际的列名选择需要的列
        available_cols = []
        desired_cols = {
            'date': ['Date', 'date', 'DATE'],
            'state': ['State', 'state', 'STATE_NAME', 'State Name'],
            'state_id': ['state_id', 'State_ID', 'STATE_ID'],
            'county': ['County', 'county', 'COUNTY_NAME', 'County Name'],
            'aqi': ['AQI', 'aqi', 'AQI Value', 'Overall AQI Value'],
            'category': ['Category', 'category', 'CATEGORY'],
            'pm25': ['PM2.5', 'pm25', 'PM2.5 AQI Value'],
            'pm10': ['PM10', 'pm10', 'PM10 AQI Value'],
            'ozone': ['Ozone', 'ozone', 'Ozone AQI Value'],
            'city_ascii': ['city_ascii', 'City_ASCII', 'CITY_ASCII']
        }
        
        # 确定实际存在的列名
        actual_cols = {}
        for key, possible_names in desired_cols.items():
            for name in possible_names:
                if name in sample_df.columns:
                    actual_cols[key] = name
                    available_cols.append(name)
                    break
        
        # 读取完整数据
        if available_cols:
            air_quality_df = pd.read_csv(air_file, usecols=available_cols)
            
            # 重命名列以标准化
            rename_dict = {v: k for k, v in actual_cols.items() if k != 'category'}
            if 'category' in actual_cols:
                rename_dict[actual_cols['category']] = 'Category'
            if 'state_id' in actual_cols:
                rename_dict[actual_cols['state_id']] = 'state_id'
            air_quality_df = air_quality_df.rename(columns=rename_dict)
        else:
            air_quality_df = pd.read_csv(air_file)
    except Exception as e:
        print(f"   读取空气质量数据时出错: {e}")
        return None
    
    # 筛选加州数据
    if 'state_id' in air_quality_df.columns:
        # 用state_id筛选加州
        ca_names = ['CA', 'California', 'california', 'CALIFORNIA']
        ca_air_quality = None
        for name in ca_names:
            temp_df = air_quality_df[air_quality_df['state_id'] == name]
            if len(temp_df) > 0:
                ca_air_quality = temp_df.copy()
                break
        print("   - 用state_id列筛选加州数据")
    elif 'state' in air_quality_df.columns:
        ca_names = ['CA', 'California', 'california', 'CALIFORNIA']
        ca_air_quality = None
        for name in ca_names:
            temp_df = air_quality_df[air_quality_df['state'] == name]
            if len(temp_df) > 0:
                ca_air_quality = temp_df.copy()
                break
        print("   - 用state列筛选加州数据")
    else:
        ca_air_quality = air_quality_df.copy()
    
    if ca_air_quality is None:
        print("   警告：未找到加州的空气质量数据")
        return None
    
    print(f"   - 加州空气质量记录数: {len(ca_air_quality):,}")
    print(f"   - 数据是否包含Category列: {'Category' in ca_air_quality.columns}")
    print(f"   - 数据是否包含state_id列: {'state_id' in ca_air_quality.columns}")  # 验证state_id
    
    # 处理日期
    if 'date' in ca_air_quality.columns:
        ca_air_quality['Date'] = pd.to_datetime(ca_air_quality['date'], errors='coerce')
        ca_air_quality = ca_air_quality.dropna(subset=['Date'])
    elif 'Date' in ca_air_quality.columns:
        ca_air_quality['Date'] = pd.to_datetime(ca_air_quality['Date'], errors='coerce')
        ca_air_quality = ca_air_quality.dropna(subset=['Date'])
    
    # 处理AQI数据
    if 'aqi' in ca_air_quality.columns:
        ca_air_quality['AQI'] = pd.to_numeric(ca_air_quality['aqi'], errors='coerce')
    elif 'AQI' in ca_air_quality.columns:
        ca_air_quality['AQI'] = pd.to_numeric(ca_air_quality['AQI'], errors='coerce')
    
    return ca_air_quality

# 以下是你原来的调用和检查代码
ca_accidents = load_accident_data()
ca_air_quality = load_air_quality_data()

# ------------------- 1.4 检查数据是否加载成功 -------------------
if ca_accidents is None:
    print("\n事故真实数据加载失败！请先解决上述事故数据相关错误（如文件路径、列缺失）。")
    exit()
if ca_air_quality is None:
    print("\n空气质量真实数据加载失败！请先解决上述空气质量数据相关错误。")
    exit()


print(f"\n数据加载完成:")
print(f"  - 交通事故记录: {len(ca_accidents):,}")
print(f"  - 空气质量记录: {len(ca_air_quality):,}")
print(f"\n数据加载完成:")
print(f"  - 交通事故记录: {len(ca_accidents):,}")
print(f"  - 空气质量记录: {len(ca_air_quality):,}")

# ==================== 2. 数据探索性分析 ====================
print("\n2. 数据探索性分析")
print("-" * 40)

# 2.1 数据基本信息
print("\n2.1 交通事故数据概览:")
print(f"   时间范围: {ca_accidents['Start_Time'].min()} 至 {ca_accidents['Start_Time'].max()}")
print(f"   覆盖城市数: {ca_accidents['City'].nunique()}")

# 2.2 交通事故严重程度分布
print("\n2.2 交通事故严重程度分布:")
severity_counts = ca_accidents['Severity'].value_counts().sort_index()
for sev, count in severity_counts.items():
    print(f"   严重程度 {sev}: {count:,} ({count/len(ca_accidents)*100:.1f}%)")

# 2.3 能见度统计
if 'Visibility(mi)' in ca_accidents.columns:
    visibility_stats = ca_accidents['Visibility(mi)'].dropna()
    print("\n2.3 能见度统计:")
    print(f"   平均能见度: {visibility_stats.mean():.2f} 英里")
    print(f"   中位数能见度: {visibility_stats.median():.2f} 英里")
    print(f"   最低能见度: {visibility_stats.min():.2f} 英里")
    print(f"   最高能见度: {visibility_stats.max():.2f} 英里")
    
    # 低能见度事故统计
    low_visibility = ca_accidents[ca_accidents['Visibility(mi)'] < 1.0]
    print(f"\n2.4 低能见度(<1英里)事故:")
    print(f"   数量: {len(low_visibility):,} ({len(low_visibility)/len(ca_accidents)*100:.1f}%)")
    if len(low_visibility) > 0:
        print(f"   平均严重程度: {low_visibility['Severity'].mean():.2f}")

# ==================== 3. 能见度与事故严重程度关系分析 ====================
print("\n3. 能见度与事故严重程度关系分析")
print("-" * 40)

if 'Visibility(mi)' in ca_accidents.columns:
    ca_accidents['Visibility_AQI_Category'] = pd.cut(
        ca_accidents['Visibility(mi)'], 
        bins=[0, 1, 3, 5, 10, float('inf')],
        labels=['极低(<1)', '低(1-3)', '中(3-5)', '良好(5-10)', '优秀(>10)']
    )
    
    visibility_severity = ca_accidents.groupby('Visibility_AQI_Category')['Severity'].agg(['mean', 'count'])
    print("\n能见度类别与平均事故严重程度:")
    for idx, row in visibility_severity.iterrows():
        if row['count'] > 0:
            print(f"   {idx}: 平均严重程度={row['mean']:.2f}, 事故数={row['count']:,}")
    
    # 相关性分析
    valid_data = ca_accidents[['Visibility(mi)', 'Severity']].dropna()
    if len(valid_data) > 0:
        correlation = valid_data.corr().iloc[0, 1]
        print(f"\n能见度与事故严重程度相关系数: {correlation:.3f}")

# ==================== 4. 时空分布分析 ====================
print("\n4. 时空分布分析")
print("-" * 40)

# 4.1 月度事故分布
monthly_accidents = ca_accidents.groupby('Month').size()
print("\n4.1 月度事故分布:")
month_names = ['一月', '二月', '三月', '四月', '五月', '六月', 
               '七月', '八月', '九月', '十月', '十一月', '十二月']
for month, count in monthly_accidents.items():
    if 1 <= month <= 12:
        print(f"   {month_names[month-1]}: {count:,} 起")

# 4.2 小时事故分布
hourly_accidents = ca_accidents.groupby('Hour').size()
peak_hour = hourly_accidents.idxmax()
print(f"\n4.2 事故高峰时段: {peak_hour}:00 ({hourly_accidents[peak_hour]:,} 起)")

# 4.3 工作日vs周末
ca_accidents['IsWeekend'] = ca_accidents['DayOfWeek'].isin([5, 6])
weekend_stats = ca_accidents.groupby('IsWeekend')['Severity'].agg(['count', 'mean'])
print("\n4.3 工作日 vs 周末:")
print(f"   工作日事故: {weekend_stats.loc[False, 'count']:,} 起, 平均严重度: {weekend_stats.loc[False, 'mean']:.2f}")
if True in weekend_stats.index:
    print(f"   周末事故: {weekend_stats.loc[True, 'count']:,} 起, 平均严重度: {weekend_stats.loc[True, 'mean']:.2f}")

# 4.4 城市事故分布
city_accidents = ca_accidents['City'].value_counts().head(10)
print("\n4.4 事故最多的前10个城市:")
for i, (city, count) in enumerate(city_accidents.items(), 1):
    print(f"   {i}. {city}: {count:,} 起")

# ==================== 5. 空气质量数据分析 ====================
if ca_air_quality is not None and ('AQI' in ca_air_quality.columns or 'aqi' in ca_air_quality.columns):
    print("\n5. 空气质量数据分析")
    print("-" * 40)
    
    # 确保AQI列存在且为数值型
    if 'aqi' in ca_air_quality.columns and 'AQI' not in ca_air_quality.columns:
        ca_air_quality['AQI'] = ca_air_quality['aqi']
    aqi_stats = ca_air_quality['AQI'].dropna()
    
    if len(aqi_stats) > 0:
        print(f"\n5.1 AQI统计:")
        print(f"   平均AQI: {aqi_stats.mean():.1f}")
        print(f"   最高AQI: {aqi_stats.max():.1f}")
        print(f"   最低AQI: {aqi_stats.min():.1f}")
        
        # AQI分类（统一列名为AQI_Category）
        ca_air_quality['AQI_Category'] = pd.cut(
            ca_air_quality['AQI'],
            bins=[0, 50, 100, 150, 200, 300, float('inf')],
            labels=['良好', '中等', '对敏感人群不健康', '不健康', '非常不健康', '有害']
        )
        
        aqi_distribution = ca_air_quality['AQI_Category'].value_counts()
        print("\n5.2 AQI分类分布:")
        for category, count in aqi_distribution.items():
            print(f"   {category}: {count:,} 天")
    print(f"   - 生成后是否有AQI_Category列: {'AQI_Category' in ca_air_quality.columns}")

# ==================== 6. 数据关联分析 ====================
print("\n6. 空气质量与交通事故关联分析")
print("-" * 40)

# 尝试关联两个数据集
if 'Date' in ca_accidents.columns and ca_air_quality is not None and 'Date' in ca_air_quality.columns:
    # 确保两个数据集的Date列都是datetime类型
    ca_accidents['Date'] = pd.to_datetime(ca_accidents['Date'])
    ca_air_quality['Date'] = pd.to_datetime(ca_air_quality['Date'])
    
    # 按日期和城市聚合事故数据
    daily_accidents = ca_accidents.groupby(['Date', 'City']).agg({
        'Severity': 'mean',
        'ID': 'count'
    }).rename(columns={'ID': 'Accident_Count'}).reset_index()
    
    # 尝试合并数据
    if 'City' in ca_air_quality.columns:
        # 确保Date列类型一致
        daily_accidents['Date'] = pd.to_datetime(daily_accidents['Date'])
        ca_air_quality_subset = ca_air_quality[['Date', 'City', 'AQI']].dropna()
        ca_air_quality_subset['Date'] = pd.to_datetime(ca_air_quality_subset['Date'])
        
        merged_data = pd.merge(
            daily_accidents,
            ca_air_quality_subset,
            on=['Date', 'City'],
            how='inner'
        )
    else:
        # 如果空气质量数据没有城市信息，只按日期合并
        daily_air = ca_air_quality.groupby('Date')['AQI'].mean().reset_index()
        daily_air['Date'] = pd.to_datetime(daily_air['Date']) 
        
        merged_data = pd.merge(daily_accidents, daily_air, on='Date', how='inner')
        merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    
    if len(merged_data) > 0:
        print(f"\n成功关联的记录数: {len(merged_data):,}")
        
        # 计算AQI与事故的相关性
        if 'AQI' in merged_data.columns:
            # 确保数据是数值类型
            merged_data['AQI'] = pd.to_numeric(merged_data['AQI'], errors='coerce')
            merged_data['Severity'] = pd.to_numeric(merged_data['Severity'], errors='coerce')
            merged_data['Accident_Count'] = pd.to_numeric(merged_data['Accident_Count'], errors='coerce')
            
            # 删除包含NaN的行
            clean_data = merged_data[['AQI', 'Severity', 'Accident_Count']].dropna()
            
            if len(clean_data) > 1:
                corr_severity = clean_data[['AQI', 'Severity']].corr().iloc[0, 1]
                corr_count = clean_data[['AQI', 'Accident_Count']].corr().iloc[0, 1]
                
                print(f"AQI与事故严重程度相关系数: {corr_severity:.3f}")
                print(f"AQI与事故数量相关系数: {corr_count:.3f}")
                
                # 添加更详细的分析
                if corr_severity > 0.1:
                    print("   → 发现正相关：空气质量越差(AQI越高)，事故严重程度越高")
                elif corr_severity < -0.1:
                    print("   → 发现负相关：空气质量越差(AQI越高)，事故严重程度越低")
                else:
                    print("   → 相关性较弱")
            else:
                print("   合并后的有效数据不足，无法计算相关性")
    else:
        print("   未能成功关联数据，可能是日期范围不重叠")
else:
    print("   缺少必要的日期或空气质量数据，无法进行关联分析")
    


# ==================== 7. 可视化分析 ====================
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

print("\n7. 生成独立可视化图表...")

current_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = current_dir
print(f"\n图表将保存到：{SAVE_DIR}")

def get_save_path(filename):
    return os.path.join(SAVE_DIR, filename)
    
simhei_path = "C:\\Windows\\Fonts\\simhei.ttf"

def set_chinese_fonts(ax):
    """为图表的所有中文元素统一设置 SimHei 字体"""
    # 创建字体对象
    title_font = FontProperties(fname=simhei_path, size=18, weight='bold')
    label_font = FontProperties(fname=simhei_path, size=12, weight='bold')
    legend_font = FontProperties(fname=simhei_path, size=10)
    text_font = FontProperties(fname=simhei_path, size=12)
    
    # 应用到图表
    ax.title.set_fontproperties(title_font)
    ax.xaxis.label.set_fontproperties(label_font)  # x轴标签
    ax.yaxis.label.set_fontproperties(label_font)  # y轴标签
    # 图例字体
    for text in ax.get_legend().get_texts() if ax.get_legend() else []:
        text.set_fontproperties(legend_font)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(label_font)
    for text in ax.texts:
        text.set_fontproperties(text_font)

# 基础配置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.unicode_minus'] = False


# ------------------- 7.1 能见度分布直方图-------------------
if 'Visibility(mi)' in ca_accidents.columns:
    fig, ax = plt.subplots(figsize=(10, 7))
    visibility_data = ca_accidents['Visibility(mi)'].dropna()
    
    if len(visibility_data) > 0:
        n, bins, patches = ax.hist(
            visibility_data, bins=30, edgecolor='gray', 
            alpha=0.7, color='#87CEEB'
        )
        visibility_data.plot(
            kind='kde', ax=ax, color='#4A90E2', 
            linewidth=2
        )
        ax.axvline(
            x=1, color='#5B9BD5', linestyle='--', 
            linewidth=2
        )
        
        ax.set_title(
            '加州交通事故能见度分布（2019-2023）',
            pad=30,
            y=1.05
        )
        ax.set_xlabel('能见度（英里）')
        ax.set_ylabel('事故数量')
        ax.legend(labels=['能见度密度曲线', '低能见度阈值(1英里)'])
        ax.grid(True, alpha=0.3)
    
    else:
        ax.set_title(
            '加州交通事故能见度分布（2019-2023）',
            pad=30,
            y=1.05
        )
        ax.text(0.5, 0.5, '无有效能见度数据', ha='center', va='center')
    
    set_chinese_fonts(ax)
    
    plt.subplots_adjust(top=0.85)
    plt.savefig(get_save_path('7.1_能见度分布.png'), dpi=300, bbox_inches='tight')
    print("   ✔ 生成：7.1_能见度分布.png")
plt.close()


# ------------------- 7.2 月度事故趋势图-------------------
fig, ax = plt.subplots(figsize=(12, 6))
monthly_accidents = ca_accidents.groupby('Month').size()
month_names = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']

bars = ax.bar(month_names, monthly_accidents.values, color='#90CAF9', alpha=0.8, edgecolor='#444444', linewidth=0.8)
peak_month_idx = monthly_accidents.idxmax() - 1
bars[peak_month_idx].set_color('#42A5F5')
bars[peak_month_idx].set_edgecolor('#1976D2')
bars[peak_month_idx].set_linewidth(1.2)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 50,
            f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

y_max = monthly_accidents.max()
ax.set_ylim(0, y_max * 1.15)

ax.set_xlabel('月份')
ax.set_ylabel('事故数量')
ax.set_title('加州交通事故月度分布趋势（2019-2023）', pad=20)
ax.legend(['事故数', f'高峰月（{month_names[peak_month_idx]}, {int(y_max):,}起）'])
ax.grid(True, alpha=0.3, axis='y')

set_chinese_fonts(ax)

plt.tight_layout()
plt.savefig(get_save_path('7.2_月度事故趋势.png'), dpi=300, bbox_inches='tight')
print("   ✔ 生成：7.2_月度事故趋势.png")
plt.close()

simhei_path = "C:\\Windows\\Fonts\\simhei.ttf"
title_font = FontProperties(fname=simhei_path, size=18, weight='bold')
label_font = FontProperties(fname=simhei_path, size=12, weight='bold')
legend_font = FontProperties(fname=simhei_path, size=10)
text_font = FontProperties(fname=simhei_path, size=12)

def set_chinese_fonts(ax):
    """统一设置图表中文元素字体"""
    ax.title.set_fontproperties(title_font)
    ax.xaxis.label.set_fontproperties(label_font)
    ax.yaxis.label.set_fontproperties(label_font)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(label_font)
    for text in ax.texts:
        text.set_fontproperties(text_font)


# ------------------- 7.3 24小时事故分布（单独图表）-------------------
fig, ax = plt.subplots(figsize=(12, 6))
hourly_accidents = ca_accidents.groupby('Hour').size()
ax.plot(hourly_accidents.index, hourly_accidents.values, color='#42A5F5', linewidth=3, marker='o', markersize=6, markerfacecolor='#1976D2')

morning_peak = hourly_accidents[6:10].idxmax()
evening_peak = hourly_accidents[16:20].idxmax()
ax.scatter([morning_peak, evening_peak], [hourly_accidents[morning_peak], hourly_accidents[evening_peak]], 
           color='#1565C0', s=120, zorder=5)

ax.set_xlabel('小时（24小时制）')
ax.set_ylabel('事故数量')
ax.set_title('加州交通事故24小时分布（2019-2023）', pad=20)
ax.set_xticks(range(0, 24, 2))
ax.legend(
    labels=[f'早高峰({morning_peak}:00) / 晚高峰({evening_peak}:00)'],
    prop=legend_font,
    fontsize=10, 
    loc='upper right'
)
ax.grid(True, alpha=0.3)

# 应用通用字体设置
set_chinese_fonts(ax)
plt.tight_layout()
plt.savefig(get_save_path('7.3_24小时事故分布.png'), dpi=300, bbox_inches='tight')
print("   ✔ 生成：7.3_24小时事故分布.png")
plt.close()


# ------------------- 7.4 事故最多的10个城市（单独图表）-------------------
fig, ax = plt.subplots(figsize=(11, 8))
city_accidents = ca_accidents['City'].value_counts().head(10)

ax.set_yticks(range(len(city_accidents)))
ax.set_yticklabels(city_accidents.index, fontsize=11, ha='left')
ax.tick_params(axis='y', pad=80)

bars = ax.barh(range(len(city_accidents)), city_accidents.values, color='#90CAF9', alpha=0.8, edgecolor='#444444', linewidth=0.8, 
               height=0.6)
bars[0].set_color('#42A5F5')
bars[0].set_edgecolor('#1976D2')
bars[0].set_linewidth(1.2)

# 数值标签设置
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 250, 
            bar.get_y() + bar.get_height()/2.,
            f'{int(width):,}', ha='left', va='center', fontsize=10, fontweight='bold')

# 设置x轴范围
x_max = city_accidents.max()
ax.set_xlim(0, x_max * 1.3)

ax.set_xlabel('事故数量')
ax.set_ylabel('城市')
ax.set_title('加州事故最多的10个城市（2019-2023）', pad=20)
ax.legend(
    labels=['事故数', f'事故最多城市（{city_accidents.index[0]}, {int(x_max):,}起）'],
    prop=legend_font,
    fontsize=10, 
    loc='lower right'
)
ax.grid(True, alpha=0.3, axis='x')

set_chinese_fonts(ax)
plt.tight_layout()
plt.savefig(get_save_path('7.4_Top10事故城市.png'), dpi=300, bbox_inches='tight')
print("   ✔ 生成：7.4_Top10事故城市.png")
plt.close()


# ------------------- 7.5 事故时天气条件分布（单独图表）-------------------
if 'Weather_Condition' in ca_accidents.columns:
    fig, ax = plt.subplots(figsize=(11, 9))
    weather_counts = ca_accidents['Weather_Condition'].value_counts().head(6)
    if len(weather_counts) > 0:
        colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#1976D2']
        wedges, texts, autotexts = ax.pie(
            weather_counts.values, 
            labels=weather_counts.index, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={
                'fontproperties': text_font,
                'fontsize': 12
            }
        )
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        ax.set_title('加州交通事故时天气条件分布（前6种，2019-2023）', 
                    pad=20, fontsize=18)
        
        legend_labels = [f'{label}: {count:,}起' for label, count in weather_counts.items()]
        ax.legend(
            wedges, legend_labels, 
            loc='upper right', 
            fontsize=11,
            bbox_to_anchor=(1.25, 1.0),
            prop=legend_font
        )
    
    else:
        ax.text(0.5, 0.5, '无天气数据', 
                ha='center', va='center', 
                fontproperties=text_font, 
                fontsize=16)  # 原14
        ax.set_title('加州交通事故时天气条件分布（2019-2023）', fontsize=18)
    
    set_chinese_fonts(ax)
    plt.tight_layout()
    plt.savefig(get_save_path('7.5_事故天气分布.png'), dpi=300, bbox_inches='tight')
    print("   ✔ 生成：7.5_事故天气分布.png")
plt.close()

# ------------------- 7.6 事故严重程度分布（单独图表）-------------------
fig, ax = plt.subplots(figsize=(10, 6))
severity_counts = ca_accidents['Severity'].value_counts().sort_index()
colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#42A5F5']
bars = ax.bar(
    ['轻微(1)', '较轻(2)', '较严重(3)', '严重(4)'], 
    severity_counts.values, 
    color=colors, 
    alpha=0.8, 
    edgecolor='#444444',
    linewidth=0.8
)

total = len(ca_accidents)
for bar in bars:
    height = bar.get_height()
    percentage = (height / total) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height + 100,
            f'{int(height):,}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('事故严重程度')
ax.set_ylabel('事故数量')
ax.set_title('加州交通事故严重程度分布（2019-2023）', pad=20)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=f'严重程度{i+1}（{["轻微","较轻","较严重","严重"][i]}）') for i in range(4)]
# 图例明确指定字体
ax.legend(
    handles=legend_elements, 
    fontsize=10, 
    loc='upper right',
    prop=legend_font
)
ax.grid(True, alpha=0.3, axis='y')

# 应用通用字体设置
set_chinese_fonts(ax)
plt.tight_layout()
plt.savefig(get_save_path('7.6_事故严重程度分布.png'), dpi=300, bbox_inches='tight')
print("   ✔ 生成：7.6_事故严重程度分布.png")
plt.close()


# ------------------- 7.7 年度事故趋势（单独图表）-------------------
fig, ax = plt.subplots(figsize=(10, 6))
yearly_accidents = ca_accidents.groupby('Year').size()

bars = ax.bar(yearly_accidents.index.astype(str), yearly_accidents.values, color='#90CAF9', alpha=0.8, edgecolor='#444444', linewidth=0.8)

x = range(len(yearly_accidents))
y = yearly_accidents.values
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
ax.plot(yearly_accidents.index.astype(str), p(x), "#1976D2", linestyle='--', linewidth=2, label='年度趋势线')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 150,
            f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

y_max = yearly_accidents.max()
ax.set_ylim(0, y_max * 1.12)

ax.set_xlabel('年份')
ax.set_ylabel('事故数量')
ax.set_title('加州交通事故年度趋势（2019-2023）', pad=20)
ax.legend(prop=legend_font, fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

set_chinese_fonts(ax)
plt.tight_layout()
plt.savefig(get_save_path('7.7_年度事故趋势.png'), dpi=300, bbox_inches='tight')
print("   ✔ 生成：7.7_年度事故趋势.png")
plt.close()


# ------------------- 7.8 加州空气质量指数(AQI)分布 -------------------
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

filtered_aqi = ca_air_quality[(ca_air_quality['AQI'] >= 0) & (ca_air_quality['AQI'] <= 500)].copy()

category_colors = {
    '良好': '#E3F2FD',
    '中等': '#BBDEFB',
    '对敏感人群不健康': '#90CAF9',
    '不健康': '#42A5F5',
    '非常不健康': '#1976D2',
    '有害': '#1565C0'
}

plt.figure(figsize=(12, 8))
sns.histplot(
    data=filtered_aqi,
    x='AQI',
    hue='AQI_Category',
    palette=category_colors,
    bins=50,
    alpha=0.8
)

plt.title('加州空气质量指数(AQI)分布（2019-2023）', fontsize=16, fontweight='bold', pad=25)
plt.xlabel('空气质量指数(AQI)', fontsize=12, fontweight='bold')
plt.ylabel('频次', fontsize=12, fontweight='bold')
plt.xlim(0, 500)
plt.legend(
    handles=[Patch(facecolor=category_colors[cat], label=cat) for cat in category_colors if cat in filtered_aqi['AQI_Category'].unique()],
    title='AQI类别',
    frameon=False
)
plt.grid(axis='y', alpha=0.2)

set_chinese_fonts(plt.gca())
plt.tight_layout()
plt.savefig(get_save_path('7.8_AQI分布.png'), dpi=300, bbox_inches='tight')
print("   ✔ 生成：7.8_AQI分布.png")
plt.close()

# ------------------- 7.9 温度与能见度关系（单独图表）-------------------
if 'Temperature(F)' in ca_accidents.columns and 'Visibility(mi)' in ca_accidents.columns:
    fig, ax = plt.subplots(figsize=(10, 8))
    temp_vis_data = ca_accidents[['Temperature(F)', 'Visibility(mi)', 'Severity']].dropna()
    if len(temp_vis_data) > 0:
        sample_size = min(5000, len(temp_vis_data))
        sample_data = temp_vis_data.sample(sample_size, random_state=42)
        
        scatter = ax.scatter(
            sample_data['Temperature(F)'], 
            sample_data['Visibility(mi)'], 
            c=sample_data['Severity'],
            cmap='Blues',
            alpha=0.7, 
            s=50, 
            edgecolor='#444444',
            linewidth=0.5
        )
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(
            '事故严重程度（蓝色越深，严重程度越高：1=轻微→4=严重）', 
            fontsize=11, 
            fontweight='bold',
            fontproperties=label_font
        )
        cbar.ax.tick_params(labelsize=10)
        
        ax.set_xlim(sample_data['Temperature(F)'].min() - 5, sample_data['Temperature(F)'].max() + 5)
        ax.set_ylim(0, 12)
        
        ax.set_xlabel('温度（°F）')
        ax.set_ylabel('能见度（英里）')
        ax.set_title(f'加州交通事故：温度与能见度关系（2019-2023，样本量：{sample_size}）', pad=20)
        ax.legend(
            ['事故点（蓝色深度=严重程度）'], 
            fontsize=10, 
            loc='upper right',
            prop=legend_font
        )
        ax.grid(True, alpha=0.3)
    
    else:
        ax.text(0.5, 0.5, '无有效数据（温度/能见度缺失）', ha='center', va='center', fontproperties=text_font)
        ax.set_title('加州交通事故：温度与能见度关系（2019-2023）')
    
    set_chinese_fonts(ax)
    plt.tight_layout()
    plt.savefig(get_save_path('7.9_温度-能见度关系.png'), dpi=300, bbox_inches='tight')
    print("   ✔ 生成：7.9_温度-能见度关系.png")
plt.close()


# ------------------- 7.10 工作日vs周末事故对比（单独图表）-------------------
fig, ax = plt.subplots(figsize=(10, 7))
weekend_stats = ca_accidents.groupby('IsWeekend').agg({
    'Severity': ['mean', 'count'],
    'ID': 'count'
}).round(2)
weekend_stats.columns = ['平均严重程度', '事故数', 'ID计数']
weekend_stats.index = ['工作日', '周末']

x = np.arange(len(weekend_stats))
width = 0.4

bars = ax.bar(x, weekend_stats['事故数'], width, label='事故数', color='#90CAF9', alpha=0.8, edgecolor='#444444')
ax.set_xlabel('日期类型', fontsize=12)
ax.set_ylabel('事故数量', color='#42A5F5', fontsize=12)
ax.tick_params(axis='y', labelcolor='#42A5F5')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
            f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2 = ax.twinx()
ax2.set_ylim(1, 4)
ax2.plot(x, weekend_stats['平均严重程度'], color='#1976D2', marker='o', linewidth=3, markersize=8, 
         label='平均严重程度', markerfacecolor='#42A5F5')
ax2.set_ylabel('平均事故严重程度', color='#1976D2', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#1976D2')

for i, v in enumerate(weekend_stats['平均严重程度']):
    label_y = min(v + 0.1, 3.8)
    ax2.text(i, label_y, f'{v:.2f}', ha='center', va='bottom', 
             fontsize=10, fontweight='bold', color='#1976D2')

ax.set_xticks(x)
ax.set_xticklabels(weekend_stats.index, fontsize=11)
ax.set_title('加州工作日vs周末交通事故对比（2019-2023）', pad=20)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(
    lines1 + lines2, labels1 + labels2, 
    loc='upper right', 
    fontsize=10,
    prop=legend_font
)

set_chinese_fonts(ax)
ax2.yaxis.label.set_fontproperties(label_font)
for label in ax2.get_yticklabels():
    label.set_fontproperties(label_font)

plt.tight_layout()
plt.savefig(get_save_path('7.10_工作日vs周末事故对比.png'), dpi=300, bbox_inches='tight')
print("   ✔ 生成：7.10_工作日vs周末事故对比.png")
plt.close()


# ------------------- 7.11 能见度类别与严重程度（单独图表）-------------------
if 'Visibility_AQI_Category' in ca_accidents.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    vis_cat_stats = ca_accidents.groupby('Visibility_AQI_Category').agg({
        'Severity': ['mean', 'count']
    }).round(2)
    vis_cat_stats.columns = ['平均严重程度', '事故数']
    vis_cat_stats = vis_cat_stats[vis_cat_stats['事故数'] > 0]

    if len(vis_cat_stats) > 0:
        x = np.arange(len(vis_cat_stats))
        width = 0.5

        bars = ax.bar(x, vis_cat_stats['事故数'], width, label='事故数', color='#90CAF9', alpha=0.8, edgecolor='#444444')
        ax.set_xlabel('能见度类别')
        ax.set_ylabel('事故数量', color='#42A5F5')
        ax.tick_params(axis='y', labelcolor='#42A5F5')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax2 = ax.twinx()
        ax2.plot(x, vis_cat_stats['平均严重程度'], color='#1976D2', marker='s', linewidth=3, markersize=8, label='平均严重程度', markerfacecolor='#42A5F5')
        ax2.set_ylabel('平均事故严重程度', color='#1976D2')
        ax2.tick_params(axis='y', labelcolor='#1976D2')
        for i, v in enumerate(vis_cat_stats['平均严重程度']):
            ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1976D2')

        ax.set_xticks(x)
        ax.set_xticklabels(vis_cat_stats.index, rotation=15)
        ax.set_title('加州不同能见度类别的事故对比（2019-2023）', pad=20)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, labels1 + labels2, 
            loc='upper right', 
            fontsize=10,
            prop=legend_font
        )

    else:
        ax.text(0.5, 0.5, '无有效数据（能见度类别缺失）', ha='center', va='center', fontproperties=text_font)
        ax.set_title('加州不同能见度类别的事故对比（2019-2023）')

    set_chinese_fonts(ax)
    ax2.yaxis.label.set_fontproperties(label_font)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(label_font)

    plt.tight_layout()
    plt.savefig(get_save_path('7.11_能见度类别-事故严重程度.png'), dpi=300, bbox_inches='tight')
    print("   ✔ 生成：7.11_能见度类别-事故严重程度.png")
plt.close()


# ------------------- 7.12 AQI与每日事故数关联图-------------------
if 'Date' in ca_accidents.columns and ca_air_quality is not None and 'Date' in ca_air_quality.columns and 'AQI' in ca_air_quality.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    daily_accidents = ca_accidents.groupby('Date').size().reset_index(name='事故数')
    daily_accidents['Date'] = pd.to_datetime(daily_accidents['Date'])
    daily_aqi = ca_air_quality.groupby('Date')['AQI'].mean().reset_index(name='平均AQI')
    daily_aqi['Date'] = pd.to_datetime(daily_aqi['Date'])
    merged = pd.merge(daily_accidents, daily_aqi, on='Date', how='inner').dropna()

    if len(merged) > 0:
        ax.scatter(merged['Date'], merged['事故数'], color='#90CAF9', alpha=0.6, label='每日事故数', s=30, edgecolor='#444444', linewidth=0.3)
        ax2 = ax.twinx()
        ax2.plot(merged['Date'], merged['平均AQI'], color='#42A5F5', linewidth=2.5, label='日均AQI')
        ax2.axhline(y=100, color='#1976D2', linestyle='--', linewidth=2, label='AQI=100（不健康阈值）')
        
        ax.set_ylim(0, merged['事故数'].max() * 1.15)
        ax2.set_ylim(0, merged['平均AQI'].max() * 1.1)
        
        ax.set_xlabel('日期')
        ax.set_ylabel('每日事故数', color='#42A5F5')
        ax2.set_ylabel('日均空气质量指数(AQI)', color='#1976D2')
        ax.set_title(f'加州日均AQI与每日交通事故数关联（2019-2023，关联记录数：{len(merged)}）', pad=20)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, labels1 + labels2, 
            loc='upper right', 
            fontsize=10,
            prop=legend_font
        )
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
    
    else:
        ax.text(0.5, 0.5, '无对齐数据（日期范围不重叠）', ha='center', va='center', fontproperties=text_font)
        ax.set_title('加州日均AQI与每日交通事故数关联（2019-2023）')
    
    set_chinese_fonts(ax)
    ax2.yaxis.label.set_fontproperties(label_font)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(label_font)

    plt.tight_layout()
    plt.savefig(get_save_path('7.12_AQI-每日事故数关联.png'), dpi=300, bbox_inches='tight')
    print("   ✔ 生成：7.12_AQI-每日事故数关联.png")
plt.close()


# ------------------- 7.13 天气条件-事故严重程度对比图-------------------
if 'Weather_Condition' in ca_accidents.columns:
    fig, ax = plt.subplots(figsize=(12, 7))
    top_weather = ca_accidents['Weather_Condition'].value_counts().head(8).index
    weather_severity = ca_accidents[ca_accidents['Weather_Condition'].isin(top_weather)].groupby('Weather_Condition')['Severity'].agg(['mean', 'count']).round(2)
    weather_severity = weather_severity.sort_values('mean', ascending=False)

    if len(weather_severity) > 0:
        x = np.arange(len(weather_severity))
        width = 0.6

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(weather_severity)))
        bars = ax.bar(x, weather_severity['mean'], width, color=colors, edgecolor='#444444', linewidth=0.8)
        ax.set_xlabel('天气条件')
        ax.set_ylabel('平均事故严重程度', color='#1976D2')
        ax.tick_params(axis='y', labelcolor='#1976D2')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax2 = ax.twinx()
        ax2.plot(x, weather_severity['count'], color='#1976D2', marker='o', linewidth=3, markersize=8, label='事故数', markerfacecolor='#42A5F5')
        ax2.set_ylabel('事故数量', color='#1976D2')
        ax2.tick_params(axis='y', labelcolor='#1976D2')
        for i, v in enumerate(weather_severity['count']):
            ax2.text(i, v + 100, f'{int(v):,}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1976D2')

        ax.set_xticks(x)
        ax.set_xticklabels(weather_severity.index, rotation=45, ha='right')
        ax.set_title('加州不同天气条件下的事故严重程度对比（前8种，2019-2023）', pad=20)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#90CAF9', label='低严重程度'), Patch(facecolor='#42A5F5', label='高严重程度')]
        ax.legend(
            handles=legend_elements + ax2.get_legend_handles_labels()[0], 
            loc='upper right', 
            fontsize=10,
            prop=legend_font
        )
        ax.grid(True, alpha=0.3, axis='y')
    
    else:
        ax.text(0.5, 0.5, '无天气数据', ha='center', va='center', fontproperties=text_font)
        ax.set_title('加州不同天气条件下的事故严重程度对比（2019-2023）')

    set_chinese_fonts(ax)
    ax2.yaxis.label.set_fontproperties(label_font)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(label_font)

    plt.tight_layout()
    plt.savefig(get_save_path('7.13_天气条件-事故严重程度.png'), dpi=300, bbox_inches='tight')
    print("   ✔ 生成：7.13_天气条件-事故严重程度.png")
plt.close()


# ------------------- 7.14 能见度时段分布热力图-------------------
if 'Visibility(mi)' in ca_accidents.columns:
    fig, ax = plt.subplots(figsize=(13, 9))
    def get_time_period(hour):
        if 6 <= hour < 12:
            return '早间(6-12点)'
        elif 12 <= hour < 18:
            return '午后(12-18点)'
        elif 18 <= hour < 24:
            return '晚间(18-24点)'
        else:
            return '凌晨(0-6点)'
    ca_accidents['时段'] = ca_accidents['Hour'].apply(get_time_period)

    vis_pivot = ca_accidents.pivot_table(
        values='Visibility(mi)',
        index='IsWeekend',
        columns='时段',
        aggfunc='mean'
    ).round(2)
    vis_pivot.index = ['工作日', '周末']
    vis_pivot = vis_pivot[['凌晨(0-6点)', '早间(6-12点)', '午后(12-18点)', '晚间(18-24点)']]

    if not vis_pivot.isna().all().all():
        sns.heatmap(
            vis_pivot,
            annot=True,
            fmt='.2f',
            cmap='Blues_r',
            cbar_kws={'label': '平均能见度（英里）'},
            ax=ax,
            linewidths=0.8,
            linecolor='white',
            annot_kws={
                'fontproperties': text_font,
                'fontsize': 20,
                'fontweight': 'bold'
            }
        )
        cbar = ax.collections[0].colorbar
        cbar.set_label(
            '平均能见度（英里）',
            fontproperties=label_font,
            fontsize=14,
            fontweight='bold'
        )
        ax.set_title('加州不同日期类型×时段的平均能见度热力图（2019-2023）', 
                    pad=20, fontsize=24)
        ax.set_xlabel('时段', fontsize=20)
        ax.set_ylabel('日期类型', fontsize=20)
    
    else:
        ax.text(0.5, 0.5, '无能见度数据', 
                ha='center', va='center', 
                fontproperties=text_font, 
                fontsize=20)
        ax.set_title('加州不同日期类型×时段的平均能见度热力图（2019-2023）', fontsize=18)
    
    set_chinese_fonts(ax)
    plt.tight_layout()
    plt.savefig(get_save_path('7.14_能见度时段热力图.png'), dpi=300, bbox_inches='tight')
    print("   ✔ 生成：7.14_能见度时段热力图.png")
plt.close()


# ------------------- 7.15 Top10城市-事故严重度与AQI对比图-------------------
try:
    # 1. 确保城市列存在（兼容不同列名）
    if 'City' not in ca_accidents.columns:
        print("⚠ 警告：事故数据中没有 City 列，尝试使用 city_ascii 或 county 替代")
        possible_city_col = [c for c in ['city_ascii', 'city', 'County', 'county'] if c in ca_accidents.columns]
        if possible_city_col:
            ca_accidents.rename(columns={possible_city_col[0]: 'City'}, inplace=True)
        else:
            raise KeyError("事故数据中未找到城市或县列")

    if 'city_ascii' not in ca_air_quality.columns:
        print("⚠ 警告：空气质量数据中没有 city_ascii 列，尝试使用 city 或 County 替代")
        possible_aqi_city_col = [c for c in ['City', 'city', 'County', 'county'] if c in ca_air_quality.columns]
        if possible_aqi_city_col:
            ca_air_quality.rename(columns={possible_aqi_city_col[0]: 'city_ascii'}, inplace=True)
        else:
            raise KeyError("空气质量数据中未找到城市列")

    # 2. 计算Top10事故城市及核心指标
    top10_cities = ca_accidents['City'].value_counts().head(10).index.tolist()
    acc_city = ca_accidents[ca_accidents['City'].isin(top10_cities)].groupby('City').agg({
        'Severity': 'mean',
        'ID': 'count'
    }).round(2).reset_index()
    acc_city.columns = ['City', 'Avg_Severity', 'Accident_Count']
    aqi_city = ca_air_quality[ca_air_quality['city_ascii'].isin(top10_cities)].groupby('city_ascii').agg({
        'AQI': 'mean'
    }).round(1).reset_index()
    aqi_city.columns = ['City', 'Avg_AQI']
    
    # 3. 合并数据
    merged_city = pd.merge(acc_city, aqi_city, on='City', how='inner')
    if len(merged_city) == 0:
        raise ValueError("无匹配的城市数据，无法生成对比图")

    # 4. 绘图
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(merged_city))
    width = 0.4

    bars1 = ax.bar(x - width/2, merged_city['Avg_Severity'], width, 
                   label='平均事故严重度', color='#90CAF9', alpha=0.8, edgecolor='#444444')
    ax.set_xlabel('城市', fontproperties=label_font)
    ax.set_ylabel('平均事故严重度', color='#42A5F5', fontproperties=label_font)
    ax.tick_params(axis='y', labelcolor='#42A5F5')
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
                fontproperties=text_font)


    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, merged_city['Avg_AQI'], width,
                    label='平均AQI', color='#42A5F5', alpha=0.8, edgecolor='#1976D2')
    ax2.set_ylabel('平均空气质量指数(AQI)', color='#1976D2', fontproperties=label_font)
    ax2.tick_params(axis='y', labelcolor='#1976D2')
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        if height <= 50:
            level = "（良好）"
        elif height <= 100:
            level = "（中等）"
        else:
            level = "（不健康）"
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}{level}', ha='center', va='bottom', fontsize=10, fontweight='bold',
                fontproperties=text_font, color='#1976D2')

    ax.set_xticks(x)
    ax.set_xticklabels(merged_city['City'], rotation=45, ha='right', fontproperties=label_font)
    ax.set_title(f'加州Top10事故城市：平均严重度与AQI对比（2019-2023，有效城市数：{len(merged_city)}）', 
                 pad=20, fontproperties=title_font)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper right', fontsize=10, prop=legend_font)
    ax.grid(True, alpha=0.3, axis='y')

    set_chinese_fonts(ax)
    ax2.yaxis.label.set_fontproperties(label_font)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(label_font)

    save_path = get_save_path('7.15_Top10城市-事故与AQI.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✔ 生成：7.15_Top10城市-事故与AQI.png")

except Exception as e:
    print(f"   ✗ 7.15图生成失败：{e}")

    
# ------------------- 7.16 空间分布热力图-------------------
ca_map = folium.Map(location=[36.7783, -119.4179], zoom_start=6, tiles='CartoDB positron')

if 'Start_Lat' in ca_accidents.columns and 'Start_Lng' in ca_accidents.columns:
    sample_size = min(10000, len(ca_accidents))
    sample_accidents = ca_accidents.sample(sample_size, random_state=42)
    
    heat_data = [
        [row['Start_Lat'], row['Start_Lng']] 
        for _, row in sample_accidents.iterrows() 
        if pd.notna(row['Start_Lat']) and pd.notna(row['Start_Lng'])
    ]
    
    if len(heat_data) > 0:
        HeatMap(
            heat_data,
            radius=10,
            blur=12,
            gradient={
                0.1: 'lightgray',
                0.3: 'lime',
                0.5: 'yellow',
                0.7: 'orange',
                0.9: 'red'
            },
            max_opacity=0.8
        ).add_to(ca_map)
        
        # ---------------------- 添加文本说明 ----------------------
        # 1. 标题（固定在地图右上角）
        title_html = '''
             <h3 style="font-size:16px; color:#333; text-align:center; margin:10px">
             加州交通事故空间分布热力图（2019-2023）
             </h3>
             '''
        ca_map.get_root().html.add_child(folium.Element(title_html))
        
        # 2. 图例（固定在地图右下角）
        legend_html = '''
             <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; 
                         background-color: white; padding: 10px; border: 1px solid grey; 
                         border-radius: 5px; font-size:12px">
             <p><b>图例：事故密度</b></p>
             <p><span style="display:inline-block; width:15px; height:15px; background:lightgray; margin-right:5px"></span> 低密度</p>
             <p><span style="display:inline-block; width:15px; height:15px; background:lime; margin-right:5px"></span> 中低密度</p>
             <p><span style="display:inline-block; width:15px; height:15px; background:yellow; margin-right:5px"></span> 中等密度</p>
             <p><span style="display:inline-block; width:15px; height:15px; background:orange; margin-right:5px"></span> 中高密度</p>
             <p><span style="display:inline-block; width:15px; height:15px; background:red; margin-right:5px"></span> 高密度</p>
             </div>
             '''
        ca_map.get_root().html.add_child(folium.Element(legend_html))
        
        # 3. 解释性文字（固定在地图左下角）
        explanation_html = '''
             <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                         background-color: white; padding: 10px; border: 1px solid grey; 
                         border-radius: 5px; font-size:12px; max-width:200px">
             <p><b>说明：</b>本图基于10,000条随机采样数据绘制，颜色越深表示该区域交通事故越密集。</p>
             <p>数据来源：2019-2023年加州交通事故记录</p>
             </div>
             '''
        ca_map.get_root().html.add_child(folium.Element(explanation_html))
        # ---------------------------------------------------------
        
        save_path = get_save_path('7.16_加州事故热力图.html')
        ca_map.save(save_path)
        print(f"   ✔ 生成：7.16_加州事故热力图.html（含{len(heat_data)}个数据点）")
    else:
        print("   无有效坐标数据，无法生成热力图")
else:
    print("   缺少经纬度列（Start_Lat/Start_Lng），无法生成热力图")


# ------------------- 7.16 生成综合总图表（15张PNG，3x5布局）-------------------
print("\n7.16 生成综合总图表")
print("-" * 40)

target_nums = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
chart_paths = []
save_dir = os.path.dirname(get_save_path(''))

for num in target_nums:
    found = False
    for f in os.listdir(save_dir):
        if f.startswith(f'7.{num}_') and f.endswith('.png'):
            chart_paths.append(os.path.join(save_dir, f))
            print(f"   找到图表：{f}")
            found = True
            break
    if not found:
        placeholder = os.path.join(save_dir, f'7.{num}_缺失.png')
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, f'7.{num} 图表缺失', ha='center', va='center', fontsize=10, fontproperties=text_font)
        ax.axis('off')
        plt.savefig(placeholder, dpi=150, bbox_inches='tight')
        plt.close(fig)
        chart_paths.append(placeholder)
        print(f"   ⚠ 7.{num} 图表缺失，已生成占位图")

plt.figure(figsize=(25, 15))
for i in range(len(chart_paths)):
    ax = plt.subplot(3, 5, i + 1)
    path = chart_paths[i]
    
    try:
        img = mpimg.imread(path)
        ax.imshow(img)
    except Exception as e:
        ax.text(0.5, 0.5, f'加载失败\n{str(e)[:10]}...', ha='center', va='center', fontsize=8)
    
    ax.axis('off')
    ax.set_title(f'图表 7.{target_nums[i]}', fontsize=12, pad=2, fontproperties=label_font)

# 总标题与保存
plt.suptitle('加州地区空气质量与交通事故分析综合图表（2019-2023）', fontsize=20, y=0.98, fontproperties=title_font)
plt.tight_layout(pad=1.0, rect=[0, 0, 1, 0.94])
combined_path = get_save_path('7_综合分析总图表.png')
plt.savefig(combined_path, dpi=150, bbox_inches='tight')
plt.close()

# 清理内存
if 'img' in locals():
    del img
gc.collect()
print(f"   ✔ 综合总图表已保存：{combined_path}")

try:
    if os.name == 'nt':
        os.startfile(combined_path)
    else:
        os.system(f'open "{combined_path}"' if os.name == 'posix' else f'xdg-open "{combined_path}"')
    print(f"   ✔ 已用系统软件打开综合图表")
except Exception as e:
    print(f"   ✗ 打开图表失败：{str(e)}，可手动在目录中查看")

print("-" * 40)
print("7部分图表整合完成！")




# ==================== 8. 预测模型（基础模型，仅保留核心逻辑）====================
print("\n8. 构建事故严重程度预测基础模型")
print("-" * 40)

# 准备特征
features = []
if 'Visibility(mi)' in ca_accidents.columns:
    features.append('Visibility(mi)')
if 'Temperature(F)' in ca_accidents.columns:
    features.append('Temperature(F)')
if 'Humidity(%)' in ca_accidents.columns:
    features.append('Humidity(%)')
if 'Wind_Speed(mph)' in ca_accidents.columns:
    features.append('Wind_Speed(mph)')
features.extend(['Hour', 'Month', 'DayOfWeek'])

# 确保所有特征都存在
available_features = [f for f in features if f in ca_accidents.columns]

# 补充后续结论所需的变量
peak_hour = None
monthly_accidents = None
city_accidents = None
if 'Hour' in ca_accidents.columns:
    hourly_accidents = ca_accidents['Hour'].value_counts().sort_index()
    peak_hour = hourly_accidents.idxmax() if not hourly_accidents.empty else "N/A"
if 'Month' in ca_accidents.columns:
    monthly_accidents = ca_accidents['Month'].value_counts().sort_index()
if 'City' in ca_accidents.columns:
    city_accidents = ca_accidents['City'].value_counts().head(10)

# 基础模型
basic_model_accuracy = None
basic_feature_importance = None
if len(available_features) >= 2:
    ca_accidents['Severe'] = (ca_accidents['Severity'] >= 3).astype(int)
    model_data = ca_accidents[available_features + ['Severe']].dropna()
    
    if len(model_data['Severe'].unique()) < 2:
        print("   目标变量类别单一，无法训练分类模型")
    elif len(model_data) > 100:
        X = model_data[available_features]
        y = model_data['Severe']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 仅训练逻辑回归作为基础模型
        basic_model = LogisticRegression(random_state=42, max_iter=1000)
        basic_model.fit(X_train_scaled, y_train)
        basic_model_accuracy = basic_model.score(X_test_scaled, y_test)
        print(f"\n基础模型（逻辑回归）准确率: {basic_model_accuracy:.3f}")
        
        # 特征重要性
        basic_feature_importance = pd.DataFrame({
            'Feature': available_features,
            'Importance': abs(basic_model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        print("\n基础模型特征重要性排序:")
        for _, row in basic_feature_importance.iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.3f}")
    else:
        print("   数据量不足，无法训练基础模型")
else:
    print("   可用特征不足，无法训练基础模型")


# ==================== 9. 综合分析结论 ====================
print("\n" + "="*80)
print("分析结论总结")
print("="*80)

# 计算关键统计
summary_stats = {}
if 'Visibility(mi)' in ca_accidents.columns:
    low_vis = ca_accidents[ca_accidents['Visibility(mi)'] < 1.0]
    high_vis = ca_accidents[ca_accidents['Visibility(mi)'] > 5.0]
    if len(low_vis) > 0 and len(high_vis) > 0:
        low_vis_severe = low_vis['Severity'].mean()
        high_vis_severe = high_vis['Severity'].mean()
        severity_diff = ((low_vis_severe - high_vis_severe) / high_vis_severe) * 100 if high_vis_severe != 0 else 0
        summary_stats['low_vis_severe'] = low_vis_severe
        summary_stats['high_vis_severe'] = high_vis_severe
        summary_stats['severity_diff'] = severity_diff
        summary_stats['low_vis_count'] = len(low_vis)

# 输出总结（修复年均事故数计算）
annual_accident = len(ca_accidents) / ca_accidents['Year'].nunique() if ('Year' in ca_accidents.columns and ca_accidents['Year'].nunique() > 0) else 'N/A'

print(f"""
1. 数据概况:
   - 分析时间范围: {ca_accidents['Start_Time'].min().strftime('%Y-%m-%d') if not ca_accidents.empty else 'N/A'} 至 {ca_accidents['Start_Time'].max().strftime('%Y-%m-%d') if not ca_accidents.empty else 'N/A'}
   - 总事故记录数: {len(ca_accidents):,}
   - 覆盖城市数: {ca_accidents['City'].nunique() if 'City' in ca_accidents.columns else 'N/A'}
""")

if 'low_vis_severe' in summary_stats:
    print(f"""
2. 能见度影响分析:
   - 低能见度(<1英里)事故数: {summary_stats['low_vis_count']:,}
   - 低能见度平均事故严重程度: {summary_stats['low_vis_severe']:.2f}
   - 高能见度(>5英里)平均事故严重程度: {summary_stats['high_vis_severe']:.2f}
   - 低能见度条件下事故严重程度增加: {summary_stats['severity_diff']:.1f}%
""")

print(f"""
3. 时间模式:
   - 事故高峰时段: {peak_hour}:00
   - 事故最多月份: {monthly_accidents.idxmax() if (monthly_accidents is not None and not monthly_accidents.empty) else 'N/A'}月
   - 年均事故数: {annual_accident:.0f} 起/年
   
4. 空间分布:
   - 事故最多城市: {city_accidents.index[0] if (city_accidents is not None and not city_accidents.empty) else 'N/A'} 
     （{city_accidents.iloc[0]:,}起）
   - 前5大事故城市占比: {city_accidents.head(5).sum() / len(ca_accidents) * 100:.1f}%
""")

if basic_model_accuracy is not None:
    print(f"""
5. 基础预测模型（逻辑回归）:
   - 模型准确率: {basic_model_accuracy:.1%}
   - 最重要影响因素: {basic_feature_importance.iloc[0]['Feature'] if not basic_feature_importance.empty else 'N/A'}
""")

print("""
6. 政策建议:
   - 在低能见度条件下加强交通管制和预警
   - 重点关注早晚高峰时段的交通安全
   - 在高发城市增加安全措施
   - 冬季加强道路安全管理
   - 结合空气质量监测数据建立预警机制
""")

print("\n" + "="*80)
print("基础分析完成！")
print("="*80)



# ------------------- 输出完整中间数据-------------------
import pickle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# 保存完整数据
middle_data_path = os.path.join(current_dir, "ca_data_full.pkl")

try:
    middle_data_path = os.path.join(current_dir, "ca_data_full.pkl")
    with open(middle_data_path, 'wb') as f:
        pickle.dump({
            'ca_accidents': ca_accidents,
            'ca_air_quality': ca_air_quality,
            'raw_accident_path': os.path.join(current_dir, 'US_Accidents_March23.csv'),
            'raw_aqi_path': os.path.join(current_dir, 'US_AQI.csv'),
            'current_dir': current_dir
        }, f)
    print(f"\n✅ 已输出中间数据：{middle_data_path}")
except Exception as e:
    print(f"\n⚠ 中间数据保存失败：{str(e)}")

del ca_accidents, ca_air_quality
gc.collect()
print("\n✅ 1-9部分完成！")