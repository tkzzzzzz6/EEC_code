# 分析历史数据的真实模式
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_historical_patterns():
    """分析历史数据的真实模式"""
    print("🔍 分析历史数据的真实模式...")
    
    # 读取历史数据
    df = pd.read_csv('../PVODdatasets_v1.0/station01.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time')
    
    print(f"历史数据: {len(df)} 条记录")
    print(f"时间范围: {df['date_time'].min()} 到 {df['date_time'].max()}")
    print(f"平均功率: {df['power'].mean():.3f} MW")
    print(f"最大功率: {df['power'].max():.3f} MW")
    
    # 分析最后几天的真实模式
    last_week = df.tail(7*96)  # 最后7天
    print(f"\n📊 最后7天历史数据分析:")
    
    last_week['date'] = last_week['date_time'].dt.date
    last_week['hour'] = last_week['date_time'].dt.hour
    last_week['minute'] = last_week['date_time'].dt.minute
    
    # 每天统计
    daily_stats = last_week.groupby('date').agg({
        'power': ['mean', 'max', 'min', 'std']
    }).round(4)
    
    print("日期        | 平均功率 | 最大功率 | 最小功率 | 标准差")
    print("-" * 60)
    
    for date in sorted(last_week['date'].unique()):
        day_data = last_week[last_week['date'] == date]
        mean_power = day_data['power'].mean()
        max_power = day_data['power'].max()
        min_power = day_data['power'].min()
        std_power = day_data['power'].std()
        
        print(f"{date} | {mean_power:8.4f} | {max_power:8.4f} | {min_power:8.4f} | {std_power:8.4f}")
    
    # 分析日内模式的变化
    print(f"\n🌅 分析不同日期的日内模式差异:")
    
    # 选择最后3天进行对比
    unique_dates = sorted(last_week['date'].unique())[-3:]
    
    for i, date in enumerate(unique_dates):
        day_data = last_week[last_week['date'] == date]
        peak_hours = day_data.nlargest(5, 'power')
        print(f"\n{date} 功率峰值时段:")
        for _, row in peak_hours.iterrows():
            print(f"  {row['date_time']}: {row['power']:.3f} MW")
    
    # 分析连续性和变化模式
    print(f"\n📈 分析功率变化的连续性:")
    
    # 计算相邻时间点的功率差异
    last_week['power_diff'] = last_week['power'].diff()
    
    print(f"功率变化统计:")
    print(f"  平均变化: {last_week['power_diff'].mean():.4f} MW")
    print(f"  标准差: {last_week['power_diff'].std():.4f} MW")
    print(f"  最大增幅: {last_week['power_diff'].max():.4f} MW")
    print(f"  最大降幅: {last_week['power_diff'].min():.4f} MW")
    
    # 分析周期性模式
    print(f"\n🔄 分析周期性模式:")
    
    # 按小时统计
    hourly_stats = last_week.groupby('hour')['power'].agg(['mean', 'std', 'count'])
    
    print("UTC小时 | 平均功率 | 标准差 | 数据点数")
    print("-" * 40)
    for hour in range(24):
        if hour in hourly_stats.index:
            mean_power = hourly_stats.loc[hour, 'mean']
            std_power = hourly_stats.loc[hour, 'std']
            count = hourly_stats.loc[hour, 'count']
            print(f"{hour:2d}:00   | {mean_power:8.3f} | {std_power:6.3f} | {count:6d}")
    
    return last_week

def compare_prediction_vs_reality():
    """对比预测结果与真实模式"""
    print(f"\n🔍 对比预测结果与真实模式...")
    
    try:
        # 读取改进后的预测结果
        pred_df = pd.read_csv('results/station01_improved_7day_forecast.csv')
        pred_df['date_time'] = pd.to_datetime(pred_df['date_time'])
        pred_df['hour'] = pred_df['date_time'].dt.hour
        
        # 读取历史数据
        hist_df = pd.read_csv('../PVODdatasets_v1.0/station01.csv')
        hist_df['date_time'] = pd.to_datetime(hist_df['date_time'])
        hist_df['hour'] = hist_df['date_time'].dt.hour
        
        print(f"预测数据: {len(pred_df)} 条")
        print(f"历史数据: {len(hist_df)} 条")
        
        # 对比按小时的平均功率
        pred_hourly = pred_df.groupby('hour')['predicted_power'].mean()
        hist_hourly = hist_df.groupby('hour')['power'].mean()
        
        print(f"\n📊 按小时对比 (UTC时间):")
        print("小时 | 历史平均 | 预测平均 | 差异")
        print("-" * 40)
        
        for hour in range(24):
            if hour in hist_hourly.index and hour in pred_hourly.index:
                hist_avg = hist_hourly[hour]
                pred_avg = pred_hourly[hour]
                diff = pred_avg - hist_avg
                print(f"{hour:2d}:00 | {hist_avg:8.3f} | {pred_avg:8.3f} | {diff:+7.3f}")
        
        # 总体对比
        print(f"\n📈 总体对比:")
        print(f"历史平均功率: {hist_df['power'].mean():.3f} MW")
        print(f"预测平均功率: {pred_df['predicted_power'].mean():.3f} MW")
        print(f"历史最大功率: {hist_df['power'].max():.3f} MW")
        print(f"预测最大功率: {pred_df['predicted_power'].max():.3f} MW")
        
        # 分析问题
        ratio = pred_df['predicted_power'].mean() / hist_df['power'].mean()
        print(f"\n⚠️  问题分析:")
        print(f"预测/历史功率比值: {ratio:.3f}")
        if ratio < 0.7:
            print("❌ 预测功率明显偏低！")
        elif ratio > 1.3:
            print("❌ 预测功率明显偏高！")
        else:
            print("✅ 预测功率水平合理")
            
        # 分析每日变化
        print(f"\n📅 每日变化分析:")
        pred_df['date'] = pred_df['date_time'].dt.date
        daily_stats = pred_df.groupby('date')['predicted_power'].agg(['mean', 'max', 'std'])
        
        print("日期        | 平均功率 | 最大功率 | 标准差")
        print("-" * 50)
        for date, row in daily_stats.iterrows():
            print(f"{date} | {row['mean']:8.3f} | {row['max']:8.3f} | {row['std']:6.3f}")
            
    except Exception as e:
        print(f"❌ 对比分析失败: {e}")

if __name__ == "__main__":
    historical_data = analyze_historical_patterns()
    compare_prediction_vs_reality() 