# 调试数据集时间格式和功率分布
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

def check_data_format():
    """检查数据格式"""
    print("🔍 检查数据集格式...")
    
    # 读取数据
    df = pd.read_csv('../PVODdatasets_v1.0/station01.csv')
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查前几行
    print("\n📊 前10行数据:")
    print(df.head(10))
    
    # 转换时间
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # 检查时间范围
    print(f"\n⏰ 时间信息:")
    print(f"开始时间: {df['date_time'].min()}")
    print(f"结束时间: {df['date_time'].max()}")
    print(f"数据点数: {len(df):,}")
    
    # 分析功率分布
    print(f"\n⚡ 功率统计:")
    print(df['power'].describe())
    print(f"非零功率点数: {(df['power'] > 0).sum():,} ({(df['power'] > 0).mean()*100:.1f}%)")
    
    # 分析时间模式 - 按小时统计
    df['hour'] = df['date_time'].dt.hour
    hourly_stats = df.groupby('hour').agg({
        'power': ['mean', 'max', 'count']
    }).round(3)
    
    print(f"\n🌅 按小时统计 (UTC时间):")
    print("小时 | 平均功率 | 最大功率 | 数据点数")
    print("-" * 40)
    for hour in range(24):
        if hour in hourly_stats.index:
            mean_power = hourly_stats.loc[hour, ('power', 'mean')]
            max_power = hourly_stats.loc[hour, ('power', 'max')]
            count = hourly_stats.loc[hour, ('power', 'count')]
            print(f"{hour:2d}:00 | {mean_power:8.3f} | {max_power:8.3f} | {count:6d}")
    
    # 检查是否为UTC时间（通过功率模式判断）
    print(f"\n🌍 时间zone分析:")
    print("如果是UTC时间，中国地区的光伏发电高峰应该在UTC 2:00-8:00左右")
    
    # 找出功率最高的几个小时
    peak_hours = df.groupby('hour')['power'].mean().sort_values(ascending=False).head(6)
    print(f"功率最高的6个小时 (UTC):")
    for hour, power in peak_hours.items():
        beijing_hour = (hour + 8) % 24
        print(f"  UTC {hour:2d}:00 (北京时间 {beijing_hour:2d}:00): {power:.3f} MW")
    
    return df

if __name__ == "__main__":
    df = check_data_format() 