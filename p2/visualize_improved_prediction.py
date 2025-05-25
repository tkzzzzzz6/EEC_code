# 可视化改进后的预测效果
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def visualize_improved_prediction():
    """可视化改进后的预测效果"""
    print("📊 可视化改进后的预测效果...")
    
    # 读取历史数据
    hist_df = pd.read_csv('../PVODdatasets_v1.0/station01.csv')
    hist_df['date_time'] = pd.to_datetime(hist_df['date_time'])
    hist_df = hist_df.sort_values('date_time')
    
    # 读取改进后的预测结果
    pred_df = pd.read_csv('results/station01_improved_7day_forecast.csv')
    pred_df['date_time'] = pd.to_datetime(pred_df['date_time'])
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Station01 改进后的7天功率预测效果分析', fontsize=16, fontweight='bold')
    
    # 1. 时间序列对比
    ax1 = axes[0, 0]
    
    # 显示最后3天历史数据
    last_3days = hist_df.tail(3*96)
    ax1.plot(last_3days['date_time'], last_3days['power'], 
             label='历史数据', color='blue', alpha=0.7, linewidth=1)
    
    # 显示预测数据
    ax1.plot(pred_df['date_time'], pred_df['predicted_power'], 
             label='预测数据', color='red', alpha=0.8, linewidth=1.5)
    
    ax1.set_title('时间序列对比 (最后3天历史 + 7天预测)')
    ax1.set_xlabel('时间')
    ax1.set_ylabel('功率 (MW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # 2. 按小时平均功率对比
    ax2 = axes[0, 1]
    
    hist_df['hour'] = hist_df['date_time'].dt.hour
    pred_df['hour'] = pred_df['date_time'].dt.hour
    
    hist_hourly = hist_df.groupby('hour')['power'].mean()
    pred_hourly = pred_df.groupby('hour')['predicted_power'].mean()
    
    hours = range(24)
    ax2.plot(hours, [hist_hourly.get(h, 0) for h in hours], 
             'o-', label='历史平均', color='blue', linewidth=2)
    ax2.plot(hours, [pred_hourly.get(h, 0) for h in hours], 
             's-', label='预测平均', color='red', linewidth=2)
    
    ax2.set_title('按小时平均功率对比')
    ax2.set_xlabel('UTC小时')
    ax2.set_ylabel('平均功率 (MW)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    
    # 3. 每日功率分布
    ax3 = axes[1, 0]
    
    pred_df['date'] = pred_df['date_time'].dt.date
    daily_stats = pred_df.groupby('date')['predicted_power'].agg(['mean', 'max', 'std'])
    
    dates = daily_stats.index
    ax3.bar(range(len(dates)), daily_stats['mean'], 
            alpha=0.7, label='日平均功率', color='skyblue')
    ax3.plot(range(len(dates)), daily_stats['max'], 
             'ro-', label='日最大功率', linewidth=2)
    
    # 添加历史平均线
    hist_avg = hist_df['power'].mean()
    ax3.axhline(y=hist_avg, color='green', linestyle='--', 
                label=f'历史平均 ({hist_avg:.2f} MW)')
    
    ax3.set_title('每日功率分布')
    ax3.set_xlabel('预测日期')
    ax3.set_ylabel('功率 (MW)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(len(dates)))
    ax3.set_xticklabels([f'Day{i+1}' for i in range(len(dates))])
    
    # 4. 功率分布直方图
    ax4 = axes[1, 1]
    
    # 历史数据分布
    hist_power = hist_df[hist_df['power'] > 0]['power']
    ax4.hist(hist_power, bins=50, alpha=0.6, label='历史功率分布', 
             color='blue', density=True)
    
    # 预测数据分布
    pred_power = pred_df[pred_df['predicted_power'] > 0]['predicted_power']
    ax4.hist(pred_power, bins=30, alpha=0.6, label='预测功率分布', 
             color='red', density=True)
    
    ax4.set_title('功率分布对比')
    ax4.set_xlabel('功率 (MW)')
    ax4.set_ylabel('密度')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/improved_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print(f"\n📈 改进效果统计:")
    print(f"历史平均功率: {hist_df['power'].mean():.3f} MW")
    print(f"预测平均功率: {pred_df['predicted_power'].mean():.3f} MW")
    print(f"功率水平匹配度: {pred_df['predicted_power'].mean()/hist_df['power'].mean()*100:.1f}%")
    
    print(f"\n历史最大功率: {hist_df['power'].max():.3f} MW")
    print(f"预测最大功率: {pred_df['predicted_power'].max():.3f} MW")
    print(f"峰值水平匹配度: {pred_df['predicted_power'].max()/hist_df['power'].max()*100:.1f}%")
    
    # 每日变化分析
    daily_means = daily_stats['mean'].values
    daily_variance = np.var(daily_means)
    print(f"\n📅 每日变化分析:")
    print(f"每日平均功率方差: {daily_variance:.6f}")
    print(f"每日功率范围: {daily_means.min():.3f} - {daily_means.max():.3f} MW")
    print(f"日间差异: {(daily_means.max() - daily_means.min()):.3f} MW")
    
    return daily_stats

def compare_old_vs_new():
    """对比旧版本和新版本的预测效果"""
    print(f"\n🔄 对比旧版本和新版本的预测效果...")
    
    try:
        # 读取旧版本预测
        old_pred = pd.read_csv('results/station01_7day_forecast.csv')
        old_avg = old_pred['predicted_power'].mean()
        old_max = old_pred['predicted_power'].max()
        
        # 读取新版本预测
        new_pred = pd.read_csv('results/station01_improved_7day_forecast.csv')
        new_avg = new_pred['predicted_power'].mean()
        new_max = new_pred['predicted_power'].max()
        
        # 历史数据
        hist_df = pd.read_csv('../PVODdatasets_v1.0/station01.csv')
        hist_avg = hist_df['power'].mean()
        hist_max = hist_df['power'].max()
        
        print(f"📊 预测效果对比:")
        print(f"{'指标':<12} | {'历史数据':<10} | {'旧版本':<10} | {'新版本':<10} | {'改进幅度'}")
        print("-" * 65)
        print(f"{'平均功率':<12} | {hist_avg:8.3f}   | {old_avg:8.3f}   | {new_avg:8.3f}   | {((new_avg/hist_avg)-(old_avg/hist_avg))*100:+6.1f}%")
        print(f"{'最大功率':<12} | {hist_max:8.3f}   | {old_max:8.3f}   | {new_max:8.3f}   | {((new_max/hist_max)-(old_max/hist_max))*100:+6.1f}%")
        
        print(f"\n✅ 改进效果:")
        print(f"  平均功率匹配度: {old_avg/hist_avg*100:.1f}% → {new_avg/hist_avg*100:.1f}%")
        print(f"  最大功率匹配度: {old_max/hist_max*100:.1f}% → {new_max/hist_max*100:.1f}%")
        
    except Exception as e:
        print(f"❌ 对比失败: {e}")

if __name__ == "__main__":
    daily_stats = visualize_improved_prediction()
    compare_old_vs_new() 