# 分析每天预测结果的差异
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_daily_patterns():
    """分析每天的预测模式"""
    print("🔍 分析每天预测结果的差异...")
    
    # 读取预测结果
    df = pd.read_csv('results/station01_7day_forecast.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['date'] = df['date_time'].dt.date
    df['hour'] = df['date_time'].dt.hour
    df['minute'] = df['date_time'].dt.minute
    df['time_in_day'] = df['hour'] + df['minute'] / 60
    
    print(f"预测时间范围: {df['date_time'].min()} 到 {df['date_time'].max()}")
    print(f"总预测点数: {len(df)}")
    
    # 分析每天的统计信息
    daily_stats = df.groupby('date').agg({
        'predicted_power': ['mean', 'max', 'min', 'std']
    }).round(4)
    
    print(f"\n📊 每天预测统计:")
    print("日期        | 平均功率 | 最大功率 | 最小功率 | 标准差")
    print("-" * 60)
    
    for date in sorted(df['date'].unique()):
        day_data = df[df['date'] == date]
        mean_power = day_data['predicted_power'].mean()
        max_power = day_data['predicted_power'].max()
        min_power = day_data['predicted_power'].min()
        std_power = day_data['predicted_power'].std()
        
        print(f"{date} | {mean_power:8.4f} | {max_power:8.4f} | {min_power:8.4f} | {std_power:8.4f}")
    
    # 检查每天同一时刻的预测值
    print(f"\n🕐 检查每天同一时刻的预测值 (UTC 4:00 - 峰值时段):")
    peak_hour_data = df[df['hour'] == 4]
    for _, row in peak_hour_data.iterrows():
        print(f"  {row['date']} {row['hour']:02d}:{row['minute']:02d} - {row['predicted_power']:.4f} MW")
    
    # 分析是否每天都相同
    unique_dates = sorted(df['date'].unique())
    if len(unique_dates) >= 2:
        day1_data = df[df['date'] == unique_dates[0]].sort_values('time_in_day')
        day2_data = df[df['date'] == unique_dates[1]].sort_values('time_in_day')
        
        # 比较两天的预测值
        if len(day1_data) == len(day2_data):
            power_diff = np.abs(day1_data['predicted_power'].values - day2_data['predicted_power'].values)
            max_diff = power_diff.max()
            mean_diff = power_diff.mean()
            
            print(f"\n📈 第1天与第2天预测对比:")
            print(f"  最大差异: {max_diff:.6f} MW")
            print(f"  平均差异: {mean_diff:.6f} MW")
            
            if max_diff < 1e-6:
                print("  ❌ 问题确认: 每天的预测值完全相同！")
            else:
                print("  ✅ 每天的预测值有差异")
    
    return df

def check_feature_importance():
    """检查特征重要性，看是否过度依赖时间特征"""
    try:
        import joblib
        
        # 加载模型
        model_data = joblib.load('results/station01_xgboost_model.pkl')
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # 获取特征重要性
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 特征重要性分析 (Top 10):")
        print("特征名称                    | 重要性")
        print("-" * 45)
        
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            feature_type = "时间特征" if any(x in row['feature'] for x in ['hour', 'day', 'month', 'time_slot']) else "功率特征"
            print(f"{row['feature']:<25} | {row['importance']:.4f} ({feature_type})")
        
        # 统计特征类型占比
        time_features = feature_importance[feature_importance['feature'].str.contains('hour|day|month|time_slot|sin|cos')]
        power_features = feature_importance[feature_importance['feature'].str.contains('power_')]
        
        time_importance = time_features['importance'].sum()
        power_importance = power_features['importance'].sum()
        
        print(f"\n📊 特征类型重要性占比:")
        print(f"  时间特征总重要性: {time_importance:.4f} ({time_importance/(time_importance+power_importance)*100:.1f}%)")
        print(f"  功率特征总重要性: {power_importance:.4f} ({power_importance/(time_importance+power_importance)*100:.1f}%)")
        
        if time_importance > power_importance:
            print("  ⚠️  问题发现: 模型过度依赖时间特征，缺乏历史功率的连续性！")
        
    except Exception as e:
        print(f"❌ 无法加载模型: {e}")

if __name__ == "__main__":
    df = analyze_daily_patterns()
    check_feature_importance() 