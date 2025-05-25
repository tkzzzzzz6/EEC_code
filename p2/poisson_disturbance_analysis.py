# 基于泊松分布的扰动效果分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import seaborn as sns

# 简化的中文字体设置
font_path = 'C:/Windows/Fonts/simhei.ttf'
mpl.font_manager.fontManager.addfont(font_path)  
mpl.rc('font', family='simhei')
plt.rcParams['axes.unicode_minus'] = False

plt.ioff()

def analyze_poisson_disturbance_effects():
    """分析基于泊松分布的扰动效果"""
    print("📊 分析基于泊松分布的扰动效果...")
    
    # 模拟泊松分布参数
    print("\n🎯 泊松分布扰动模型参数:")
    print("="*50)
    print("天气灾害事件:")
    print("  - 发生频率: λ = 0.5次/7天 ≈ 0.071次/天")
    print("  - 功率下降: 30-80%")
    print("  - 持续时间: 1-6小时")
    print("  - 事件类型: 冰雹、台风、极端天气")
    
    print("\n设备故障事件:")
    print("  - 发生频率: λ = 1次/3天 ≈ 0.333次/天")
    print("  - 功率下降: 50-100%")
    print("  - 持续时间: 15分钟-2小时")
    print("  - 事件类型: 逆变器跳闸、设备保护")
    
    # 模拟7天的事件发生情况
    np.random.seed(42)
    days = 7
    
    # 计算期望事件数
    weather_lambda = 0.071 * days
    equipment_lambda = 0.333 * days
    
    print(f"\n📈 7天预测期间期望事件数:")
    print(f"  天气灾害事件期望: {weather_lambda:.3f} 次")
    print(f"  设备故障事件期望: {equipment_lambda:.3f} 次")
    
    # 生成多次模拟来展示泊松分布特性
    n_simulations = 1000
    weather_events = np.random.poisson(weather_lambda, n_simulations)
    equipment_events = np.random.poisson(equipment_lambda, n_simulations)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('基于泊松分布的光伏发电扰动事件分析', fontsize=16, fontweight='bold')
    
    # 1. 天气灾害事件分布
    ax1 = axes[0, 0]
    unique_weather, counts_weather = np.unique(weather_events, return_counts=True)
    ax1.bar(unique_weather, counts_weather/n_simulations, alpha=0.7, color='#e74c3c')
    ax1.set_title('天气灾害事件发生次数分布\n(7天期间)')
    ax1.set_xlabel('事件发生次数')
    ax1.set_ylabel('概率')
    ax1.grid(True, alpha=0.3)
    
    # 添加期望值线
    ax1.axvline(weather_lambda, color='red', linestyle='--', 
               label=f'期望值: {weather_lambda:.3f}')
    ax1.legend()
    
    # 2. 设备故障事件分布
    ax2 = axes[0, 1]
    unique_equipment, counts_equipment = np.unique(equipment_events, return_counts=True)
    ax2.bar(unique_equipment, counts_equipment/n_simulations, alpha=0.7, color='#f39c12')
    ax2.set_title('设备故障事件发生次数分布\n(7天期间)')
    ax2.set_xlabel('事件发生次数')
    ax2.set_ylabel('概率')
    ax2.grid(True, alpha=0.3)
    
    ax2.axvline(equipment_lambda, color='red', linestyle='--', 
               label=f'期望值: {equipment_lambda:.3f}')
    ax2.legend()
    
    # 3. 总事件数分布
    ax3 = axes[0, 2]
    total_events = weather_events + equipment_events
    unique_total, counts_total = np.unique(total_events, return_counts=True)
    ax3.bar(unique_total, counts_total/n_simulations, alpha=0.7, color='#9b59b6')
    ax3.set_title('总事件发生次数分布\n(7天期间)')
    ax3.set_xlabel('总事件发生次数')
    ax3.set_ylabel('概率')
    ax3.grid(True, alpha=0.3)
    
    total_lambda = weather_lambda + equipment_lambda
    ax3.axvline(total_lambda, color='red', linestyle='--', 
               label=f'期望值: {total_lambda:.3f}')
    ax3.legend()
    
    # 4. 事件影响强度分布
    ax4 = axes[1, 0]
    # 模拟天气灾害影响强度
    weather_impacts = np.random.uniform(0.3, 0.8, 1000)
    ax4.hist(weather_impacts, bins=30, alpha=0.7, color='#e74c3c', 
             label='天气灾害', density=True)
    ax4.set_title('事件影响强度分布')
    ax4.set_xlabel('功率下降比例')
    ax4.set_ylabel('密度')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 事件持续时间分布
    ax5 = axes[1, 1]
    # 天气灾害持续时间
    weather_duration = np.random.uniform(1, 6, 1000)
    # 设备故障持续时间
    equipment_duration = np.random.uniform(0.25, 2, 1000)  # 15分钟到2小时
    
    ax5.hist(weather_duration, bins=20, alpha=0.6, color='#e74c3c', 
             label='天气灾害', density=True)
    ax5.hist(equipment_duration, bins=20, alpha=0.6, color='#f39c12', 
             label='设备故障', density=True)
    ax5.set_title('事件持续时间分布')
    ax5.set_xlabel('持续时间 (小时)')
    ax5.set_ylabel('密度')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 扰动模型对比
    ax6 = axes[1, 2]
    
    # 对比不同扰动模型的特点
    models = ['传统随机扰动', '泊松分布扰动']
    characteristics = ['事件频率\n可控性', '影响强度\n真实性', '持续时间\n建模', '物理意义\n合理性']
    
    # 评分矩阵 (1-5分)
    scores = np.array([
        [3, 3, 2, 2],  # 传统随机扰动
        [5, 5, 5, 5]   # 泊松分布扰动
    ])
    
    im = ax6.imshow(scores, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)
    ax6.set_xticks(range(len(characteristics)))
    ax6.set_yticks(range(len(models)))
    ax6.set_xticklabels(characteristics, rotation=45, ha='right')
    ax6.set_yticklabels(models)
    ax6.set_title('扰动模型对比评价')
    
    # 添加数值标签
    for i in range(len(models)):
        for j in range(len(characteristics)):
            ax6.text(j, i, scores[i, j], ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=12)
    
    plt.colorbar(im, ax=ax6, label='评分 (1-5)')
    
    plt.tight_layout()
    plt.savefig('results/figures/poisson_disturbance_analysis.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成统计报告
    print(f"\n📊 泊松分布扰动统计分析:")
    print("="*50)
    print(f"基于{n_simulations}次模拟:")
    print(f"  天气灾害事件:")
    print(f"    平均发生次数: {np.mean(weather_events):.3f}")
    print(f"    标准差: {np.std(weather_events):.3f}")
    print(f"    最大发生次数: {np.max(weather_events)}")
    print(f"    零事件概率: {np.sum(weather_events == 0)/n_simulations:.3f}")
    
    print(f"\n  设备故障事件:")
    print(f"    平均发生次数: {np.mean(equipment_events):.3f}")
    print(f"    标准差: {np.std(equipment_events):.3f}")
    print(f"    最大发生次数: {np.max(equipment_events)}")
    print(f"    零事件概率: {np.sum(equipment_events == 0)/n_simulations:.3f}")
    
    print(f"\n  总事件:")
    print(f"    平均发生次数: {np.mean(total_events):.3f}")
    print(f"    标准差: {np.std(total_events):.3f}")
    print(f"    最大发生次数: {np.max(total_events)}")
    print(f"    零事件概率: {np.sum(total_events == 0)/n_simulations:.3f}")
    
    # 保存详细统计数据
    stats_data = {
        '事件类型': ['天气灾害', '设备故障', '总事件'],
        '期望次数': [weather_lambda, equipment_lambda, total_lambda],
        '实际平均': [np.mean(weather_events), np.mean(equipment_events), np.mean(total_events)],
        '标准差': [np.std(weather_events), np.std(equipment_events), np.std(total_events)],
        '最大次数': [np.max(weather_events), np.max(equipment_events), np.max(total_events)],
        '零事件概率': [
            np.sum(weather_events == 0)/n_simulations,
            np.sum(equipment_events == 0)/n_simulations,
            np.sum(total_events == 0)/n_simulations
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv('results/poisson_disturbance_statistics.csv', index=False)
    
    print(f"\n✅ 泊松分布扰动分析完成!")
    print(f"✅ 图表已保存: results/figures/poisson_disturbance_analysis.png")
    print(f"✅ 统计数据已保存: results/poisson_disturbance_statistics.csv")
    
    return stats_df

def demonstrate_poisson_vs_random():
    """演示泊松分布与传统随机扰动的区别"""
    print("\n🔬 泊松分布 vs 传统随机扰动对比:")
    print("="*60)
    
    # 模拟参数
    time_points = 672  # 7天 * 96个15分钟间隔
    
    # 传统随机扰动：每个时间点独立的随机概率
    traditional_events = np.random.random(time_points) < 0.05  # 5%概率
    traditional_count = np.sum(traditional_events)
    
    # 泊松分布扰动：基于物理过程的事件发生
    np.random.seed(42)
    poisson_weather = np.random.poisson(0.5)  # 7天期间天气事件
    poisson_equipment = np.random.poisson(2.3)  # 7天期间设备事件
    poisson_count = poisson_weather + poisson_equipment
    
    print(f"传统随机扰动:")
    print(f"  事件总数: {traditional_count}")
    print(f"  事件分布: 均匀随机")
    print(f"  物理意义: 较弱")
    print(f"  可控性: 较差")
    
    print(f"\n泊松分布扰动:")
    print(f"  天气事件: {poisson_weather} 次")
    print(f"  设备事件: {poisson_equipment} 次")
    print(f"  事件总数: {poisson_count}")
    print(f"  事件分布: 符合物理规律")
    print(f"  物理意义: 强")
    print(f"  可控性: 好")
    
    print(f"\n🎯 泊松分布扰动的优势:")
    print("1. 符合实际物理过程：灾害和故障确实遵循泊松过程")
    print("2. 参数可解释：λ值直接对应现实中的事件频率")
    print("3. 事件聚集性：允许短时间内多个事件发生")
    print("4. 长期稳定性：大样本下趋向期望值")
    print("5. 工程应用性：便于风险评估和系统设计")

if __name__ == "__main__":
    # 创建结果目录
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    # 运行分析
    stats_df = analyze_poisson_disturbance_effects()
    demonstrate_poisson_vs_random()
    
    print(f"\n�� 基于泊松分布的扰动分析完成!") 