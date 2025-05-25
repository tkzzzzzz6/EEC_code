# 数据探索脚本 - 分析PVOD数据集
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def explore_pvod_data():
    """探索PVOD数据集"""
    data_dir = Path("../PVODdatasets_v1.0")
    
    # 读取元数据
    metadata = pd.read_csv(data_dir / "metadata.csv")
    print("📊 元数据信息:")
    print(f"站点数量: {len(metadata)}")
    print(f"总装机容量: {metadata['Capacity'].sum()/1000:.1f} MW")
    print("\n站点基本信息:")
    print(metadata[['Station_ID', 'Capacity', 'PV_Technology', 'Longitude', 'Latitude']].to_string(index=False))
    
    # 分析单个站点数据
    station_id = "station01"
    print(f"\n🔍 分析 {station_id} 数据:")
    
    df = pd.read_csv(data_dir / f"{station_id}.csv")
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['date_time'].min()} 到 {df['date_time'].max()}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 转换时间列
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # 分析功率数据
    print(f"\n⚡ 功率数据分析:")
    print(f"功率范围: {df['power'].min():.3f} - {df['power'].max():.3f} MW")
    print(f"平均功率: {df['power'].mean():.3f} MW")
    print(f"非零功率记录: {(df['power'] > 0).sum()} / {len(df)} ({(df['power'] > 0).mean()*100:.1f}%)")
    
    # 分析时间间隔
    df_sorted = df.sort_values('date_time')
    time_diff = df_sorted['date_time'].diff().dropna()
    print(f"\n⏰ 时间间隔分析:")
    print(f"时间间隔: {time_diff.mode().iloc[0]}")
    print(f"数据频率: 每{time_diff.mode().iloc[0].total_seconds()/60:.0f}分钟一个数据点")
    
    # 检查数据完整性
    print(f"\n📋 数据质量:")
    missing_data = df.isnull().sum()
    print("缺失值统计:")
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    # 分析功率的时间模式
    df['hour'] = df['date_time'].dt.hour
    df['month'] = df['date_time'].dt.month
    df['dayofyear'] = df['date_time'].dt.dayofyear
    
    # 日内功率模式
    hourly_power = df.groupby('hour')['power'].mean()
    print(f"\n🌅 日内功率模式 (平均功率 MW):")
    for hour in range(0, 24, 3):
        print(f"  {hour:02d}:00 - {hour+2:02d}:59: {hourly_power[hour:hour+3].mean():.3f}")
    
    return df, metadata

def analyze_prediction_feasibility(df):
    """分析预测可行性"""
    print(f"\n🎯 预测可行性分析:")
    
    # 计算自相关性
    power_series = df.set_index('date_time')['power'].resample('15T').mean()
    
    # 分析周期性
    print(f"数据点总数: {len(power_series)}")
    print(f"每天数据点数: {24*4} (15分钟间隔)")
    print(f"一周数据点数: {7*24*4}")
    
    # 分析数据的季节性和趋势
    daily_power = power_series.resample('D').mean()
    print(f"日均功率变化范围: {daily_power.min():.3f} - {daily_power.max():.3f} MW")
    print(f"日均功率标准差: {daily_power.std():.3f} MW")
    
    return power_series

if __name__ == "__main__":
    print("🚀 PVOD数据集探索分析")
    print("="*60)
    
    df, metadata = explore_pvod_data()
    power_series = analyze_prediction_feasibility(df)
    
    print(f"\n✅ 数据探索完成!")
    print(f"📁 建议使用 {df.shape[0]} 条记录进行模型训练")
    print(f"🎯 预测目标: 基于历史功率预测未来7天的15分钟级功率") 