# 查看预测结果
import pandas as pd
import numpy as np
from pathlib import Path

def check_forecast_results():
    """检查预测结果"""
    results_dir = Path("results")
    
    # 查看所有预测文件
    forecast_files = list(results_dir.glob("*_7day_forecast.csv"))
    
    print("🔍 预测结果分析")
    print("="*60)
    
    for file in forecast_files:
        station_id = file.stem.replace("_7day_forecast", "")
        print(f"\n📊 {station_id} 预测结果:")
        
        df = pd.read_csv(file)
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        print(f"  时间范围: {df['date_time'].min()} 到 {df['date_time'].max()}")
        print(f"  预测点数: {len(df):,}")
        print(f"  平均功率: {df['predicted_power'].mean():.3f} MW")
        print(f"  最大功率: {df['predicted_power'].max():.3f} MW")
        print(f"  最小功率: {df['predicted_power'].min():.3f} MW")
        print(f"  非零功率点数: {(df['predicted_power'] > 0).sum()} ({(df['predicted_power'] > 0).mean()*100:.1f}%)")
        
        # 分析日内模式
        df['hour'] = df['date_time'].dt.hour
        hourly_avg = df.groupby('hour')['predicted_power'].mean()
        peak_hour = hourly_avg.idxmax()
        peak_power = hourly_avg.max()
        
        print(f"  峰值时段: {peak_hour}:00, 平均功率: {peak_power:.3f} MW")
        
        # 显示前几个预测值
        print(f"  前5个预测值:")
        for i in range(min(5, len(df))):
            time = df.iloc[i]['date_time']
            power = df.iloc[i]['predicted_power']
            print(f"    {time}: {power:.3f} MW")

def check_comparison_results():
    """检查对比结果"""
    comparison_file = Path("results/stations_prediction_comparison.csv")
    
    if comparison_file.exists():
        print(f"\n📋 多站点对比结果:")
        print("="*60)
        
        df = pd.read_csv(comparison_file)
        print(df.to_string(index=False, float_format='%.4f'))
    
    # 检查摘要报告
    report_file = Path("results/prediction_summary_report.md")
    if report_file.exists():
        print(f"\n📄 摘要报告已生成: {report_file}")
        print(f"文件大小: {report_file.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    check_forecast_results()
    check_comparison_results()
    
    print(f"\n✅ 结果检查完成！")
    print(f"📁 所有文件保存在 results/ 目录下")
    print(f"🎨 可视化图表保存在 results/figures/ 目录下") 