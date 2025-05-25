# 光伏发电功率预测评价指标计算 + 可视化
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体和绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PowerPredictionEvaluator:
    """光伏发电功率预测评价指标计算器 + 可视化"""
    
    def __init__(self, capacity):
        """
        初始化评价器
        
        Args:
            capacity: 开机容量 (MW)
        """
        self.capacity = capacity
    
    def calculate_metrics(self, actual_power, predicted_power, only_daytime=True):
        """
        计算评价指标
        
        Args:
            actual_power: 实际功率数组
            predicted_power: 预测功率数组
            only_daytime: 是否只计算白天时段指标
            
        Returns:
            dict: 包含所有评价指标的字典
        """
        # 确保数组长度一致
        min_len = min(len(actual_power), len(predicted_power))
        actual = np.array(actual_power[:min_len])
        predicted = np.array(predicted_power[:min_len])
        
        if only_daytime:
            # 只计算白天时段（功率大于0的时段）
            daytime_mask = (actual > 0) | (predicted > 0)
            if np.sum(daytime_mask) == 0:
                return {}
            actual = actual[daytime_mask]
            predicted = predicted[daytime_mask]
        
        n = len(actual)
        
        # 归一化误差（相对于开机容量）
        normalized_actual = actual / self.capacity
        normalized_predicted = predicted / self.capacity
        normalized_error = normalized_predicted - normalized_actual
        
        # 1. 均方根误差 (RMSE)
        rmse = np.sqrt(np.mean(normalized_error ** 2))
        
        # 2. 平均绝对误差 (MAE)
        mae = np.mean(np.abs(normalized_error))
        
        # 3. 平均误差 (ME)
        me = np.mean(normalized_error)
        
        # 4. 相关系数 (r)
        if n > 1 and np.std(actual) > 0 and np.std(predicted) > 0:
            correlation = np.corrcoef(actual, predicted)[0, 1]
        else:
            correlation = 0
        
        # 5. 准确率 (CR)
        accuracy = (1 - rmse) * 100
        
        # 6. 合格率 (QR) - 误差小于25%的比例
        qualified_mask = np.abs(normalized_error) < 0.25
        qualification_rate = np.sum(qualified_mask) / n * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'ME': me,
            'Correlation': correlation,
            'Accuracy': accuracy,
            'Qualification_Rate': qualification_rate,
            'Sample_Count': n
        }
    
    def create_evaluation_visualization(self, actual, predicted, station_id, save_dir="results/figures"):
        """创建评价可视化图表"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建综合评价图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{station_id} 预测性能详细评价', fontsize=16, fontweight='bold')
        
        # 1. 时间序列对比
        ax1 = axes[0, 0]
        time_index = range(len(actual))
        ax1.plot(time_index, actual, label='实际功率', alpha=0.8, linewidth=1.5)
        ax1.plot(time_index, predicted, label='预测功率', alpha=0.8, linewidth=1.5)
        ax1.set_title('预测vs实际功率对比')
        ax1.set_xlabel('时间点')
        ax1.set_ylabel('功率 (MW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 散点图
        ax2 = axes[0, 1]
        ax2.scatter(actual, predicted, alpha=0.6, s=20)
        max_val = max(actual.max(), predicted.max())
        ax2.plot([0, max_val], [0, max_val], 'r--', label='理想预测线')
        ax2.set_title('预测vs实际散点图')
        ax2.set_xlabel('实际功率 (MW)')
        ax2.set_ylabel('预测功率 (MW)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 误差分布
        ax3 = axes[0, 2]
        errors = predicted - actual
        ax3.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(errors.mean(), color='red', linestyle='--', 
                   label=f'平均误差: {errors.mean():.3f}')
        ax3.set_title('预测误差分布')
        ax3.set_xlabel('预测误差 (MW)')
        ax3.set_ylabel('频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 相对误差分布
        ax4 = axes[1, 0]
        relative_errors = (predicted - actual) / self.capacity * 100
        ax4.hist(relative_errors, bins=30, alpha=0.7, edgecolor='black', color='orange')
        ax4.axvline(relative_errors.mean(), color='red', linestyle='--',
                   label=f'平均相对误差: {relative_errors.mean():.2f}%')
        ax4.set_title('相对误差分布 (相对于开机容量)')
        ax4.set_xlabel('相对误差 (%)')
        ax4.set_ylabel('频次')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 累积误差
        ax5 = axes[1, 1]
        cumulative_error = np.cumsum(errors)
        ax5.plot(time_index, cumulative_error, linewidth=2)
        ax5.set_title('累积误差趋势')
        ax5.set_xlabel('时间点')
        ax5.set_ylabel('累积误差 (MW)')
        ax5.grid(True, alpha=0.3)
        
        # 6. 误差箱线图
        ax6 = axes[1, 2]
        error_data = [errors]
        ax6.boxplot(error_data, labels=[station_id])
        ax6.set_title('预测误差箱线图')
        ax6.set_ylabel('预测误差 (MW)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{station_id}_detailed_evaluation.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_metrics_comparison_chart(self, metrics_dict, save_dir="results/figures"):
        """创建多站点指标对比图表"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        stations = list(metrics_dict.keys())
        
        # 创建指标对比图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('多站点预测性能指标对比', fontsize=16, fontweight='bold')
        
        # 1. RMSE对比
        ax1 = axes[0, 0]
        rmse_values = [metrics_dict[s]['RMSE'] for s in stations]
        bars1 = ax1.bar(stations, rmse_values, alpha=0.7, color='red')
        ax1.set_title('均方根误差 (RMSE)')
        ax1.set_ylabel('RMSE')
        ax1.grid(True, alpha=0.3)
        for bar, value in zip(bars1, rmse_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 2. MAE对比
        ax2 = axes[0, 1]
        mae_values = [metrics_dict[s]['MAE'] for s in stations]
        bars2 = ax2.bar(stations, mae_values, alpha=0.7, color='orange')
        ax2.set_title('平均绝对误差 (MAE)')
        ax2.set_ylabel('MAE')
        ax2.grid(True, alpha=0.3)
        for bar, value in zip(bars2, mae_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 3. 相关系数对比
        ax3 = axes[0, 2]
        corr_values = [metrics_dict[s]['Correlation'] for s in stations]
        bars3 = ax3.bar(stations, corr_values, alpha=0.7, color='blue')
        ax3.set_title('相关系数')
        ax3.set_ylabel('相关系数')
        ax3.grid(True, alpha=0.3)
        for bar, value in zip(bars3, corr_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 准确率对比
        ax4 = axes[1, 0]
        acc_values = [metrics_dict[s]['Accuracy'] for s in stations]
        bars4 = ax4.bar(stations, acc_values, alpha=0.7, color='green')
        ax4.set_title('准确率 (%)')
        ax4.set_ylabel('准确率 (%)')
        ax4.grid(True, alpha=0.3)
        for bar, value in zip(bars4, acc_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 5. 合格率对比
        ax5 = axes[1, 1]
        qr_values = [metrics_dict[s]['Qualification_Rate'] for s in stations]
        bars5 = ax5.bar(stations, qr_values, alpha=0.7, color='purple')
        ax5.set_title('合格率 (%)')
        ax5.set_ylabel('合格率 (%)')
        ax5.grid(True, alpha=0.3)
        for bar, value in zip(bars5, qr_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 6. 样本数对比
        ax6 = axes[1, 2]
        sample_values = [metrics_dict[s]['Sample_Count'] for s in stations]
        bars6 = ax6.bar(stations, sample_values, alpha=0.7, color='brown')
        ax6.set_title('样本数量')
        ax6.set_ylabel('样本数')
        ax6.grid(True, alpha=0.3)
        for bar, value in zip(bars6, sample_values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{value}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'multi_station_metrics_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_radar_chart(self, metrics, station_id, save_dir="results/figures"):
        """创建雷达图"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备雷达图数据
        categories = ['准确率', '合格率', '相关系数', 'RMSE(反)', 'MAE(反)', 'ME(反)']
        values = [
            metrics['Accuracy'],
            metrics['Qualification_Rate'],
            metrics['Correlation'] * 100,  # 转换为百分比
            (1 - metrics['RMSE']) * 100,   # 反向，越大越好
            (1 - metrics['MAE']) * 100,    # 反向，越大越好
            (1 - abs(metrics['ME'])) * 100  # 反向，越大越好
        ]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values += values[:1]  # 闭合图形
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label=station_id)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title(f'{station_id} 预测性能雷达图', size=16, fontweight='bold', pad=20)
        ax.grid(True)
        
        # 添加数值标签
        for angle, value, category in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, value + 5, f'{value:.1f}', ha='center', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{station_id}_radar_chart.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def evaluate_station_predictions():
    """评估所有站点的预测结果"""
    print("🔍 开始评估站点预测结果...")
    
    stations = ['station00', 'station04', 'station05', 'station09']
    all_metrics = {}
    
    # 开机容量设定（根据历史最大功率估算）
    capacities = {
        'station00': 6.628,
        'station04': 32.122,
        'station05': 42.142,
        'station09': 14.454
    }
    
    for station in stations:
        try:
            print(f"\n{'='*20} 评估 {station} {'='*20}")
            
            # 读取预测结果
            results_file = f'results/{station}_prediction_results.csv'
            if not Path(results_file).exists():
                print(f"❌ 找不到 {station} 的预测结果文件")
                continue
            
            results_df = pd.read_csv(results_file)
            actual_power = results_df['actual_power'].values
            predicted_power = results_df['predicted_power'].values
            
            # 创建评价器
            evaluator = PowerPredictionEvaluator(capacities[station])
            
            # 计算评价指标
            metrics = evaluator.calculate_metrics(actual_power, predicted_power)
            
            if not metrics:
                print(f"❌ {station} 没有有效的白天时段数据")
                continue
            
            all_metrics[station] = metrics
            
            # 打印评价指标
            print(f"\n📊 {station} 评价指标")
            print("="*50)
            print(f"样本数量: {metrics['Sample_Count']}")
            print(f"开机容量: {capacities[station]} MW")
            print(f"实际功率均值: {actual_power[actual_power > 0].mean():.3f} MW")
            print(f"预测功率均值: {predicted_power[predicted_power > 0].mean():.3f} MW")
            print(f"实际功率最大值: {actual_power.max():.3f} MW")
            print(f"预测功率最大值: {predicted_power.max():.3f} MW")
            print("-" * 50)
            print(f"1. 均方根误差 (RMSE): {metrics['RMSE']:.6f}")
            print(f"2. 平均绝对误差 (MAE): {metrics['MAE']:.6f}")
            print(f"3. 平均误差 (ME): {metrics['ME']:.6f}")
            print(f"4. 相关系数 (r): {metrics['Correlation']:.6f}")
            print(f"5. 准确率 (CR): {metrics['Accuracy']:.2f}%")
            print(f"6. 合格率 (QR): {metrics['Qualification_Rate']:.2f}%")
            
            # 创建可视化
            evaluator.create_evaluation_visualization(
                actual_power, predicted_power, station
            )
            evaluator.create_radar_chart(metrics, station)
            
            # 保存评价报告
            save_evaluation_report(station, metrics, capacities[station], 
                                 actual_power, predicted_power)
            
        except Exception as e:
            print(f"❌ 评估 {station} 时出错: {str(e)}")
            continue
    
    # 创建综合对比
    if all_metrics:
        create_comprehensive_comparison(all_metrics)
        
        # 创建综合可视化
        evaluator = PowerPredictionEvaluator(1.0)  # 临时创建用于可视化
        evaluator.create_metrics_comparison_chart(all_metrics)
    
    print(f"\n🎉 评估完成！")
    print(f"📁 评估报告保存在 results/ 目录下")
    
    return all_metrics

def save_evaluation_report(station_id, metrics, capacity, actual_power, predicted_power):
    """保存评价报告"""
    report = f"""
# {station_id} 光伏发电功率预测评价报告

## 基本信息
- **站点**: {station_id}
- **开机容量**: {capacity:.3f} MW
- **评估样本数**: {metrics['Sample_Count']}
- **实际功率均值**: {actual_power[actual_power > 0].mean():.3f} MW
- **预测功率均值**: {predicted_power[predicted_power > 0].mean():.3f} MW
- **实际功率最大值**: {actual_power.max():.3f} MW
- **预测功率最大值**: {predicted_power.max():.3f} MW

## 评价指标

### 1. 均方根误差 (RMSE)
- **数值**: {metrics['RMSE']:.6f}
- **说明**: 预测误差的均方根，越小越好

### 2. 平均绝对误差 (MAE)
- **数值**: {metrics['MAE']:.6f}
- **说明**: 预测误差的平均绝对值，越小越好

### 3. 平均误差 (ME)
- **数值**: {metrics['ME']:.6f}
- **说明**: 预测误差的平均值，接近0最好

### 4. 相关系数 (r)
- **数值**: {metrics['Correlation']:.6f}
- **说明**: 预测值与实际值的线性相关程度，越接近1越好

### 5. 准确率 (CR)
- **数值**: {metrics['Accuracy']:.2f}%
- **说明**: 基于RMSE计算的准确率，越大越好

### 6. 合格率 (QR)
- **数值**: {metrics['Qualification_Rate']:.2f}%
- **说明**: 相对误差小于25%的样本比例，越大越好

## 性能评价

"""
    
    # 性能等级评价
    if metrics['Accuracy'] >= 80:
        performance_level = "优秀"
    elif metrics['Accuracy'] >= 70:
        performance_level = "良好"
    elif metrics['Accuracy'] >= 60:
        performance_level = "一般"
    else:
        performance_level = "需要改进"
    
    report += f"- **整体性能等级**: {performance_level}\n"
    report += f"- **准确率评价**: {metrics['Accuracy']:.2f}%\n"
    report += f"- **合格率评价**: {metrics['Qualification_Rate']:.2f}%\n"
    
    # 保存报告
    with open(f'results/{station_id}_evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 评价报告已保存: results\\{station_id}_evaluation_report.md")

def create_comprehensive_comparison(all_metrics):
    """创建综合对比报告"""
    print(f"\n{'='*60}")
    print("📊 多站点预测性能综合对比")
    print("="*60)
    
    # 创建对比表格
    comparison_data = []
    for station, metrics in all_metrics.items():
        comparison_data.append({
            '站点': station,
            'RMSE': f"{metrics['RMSE']:.6f}",
            'MAE': f"{metrics['MAE']:.6f}",
            'ME': f"{metrics['ME']:.6f}",
            '相关系数': f"{metrics['Correlation']:.4f}",
            '准确率(%)': f"{metrics['Accuracy']:.2f}",
            '合格率(%)': f"{metrics['Qualification_Rate']:.2f}",
            '样本数': metrics['Sample_Count']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # 保存对比结果
    comparison_df.to_csv('results/multi_station_evaluation_comparison.csv', index=False)
    
    # 生成综合报告（使用简单的表格格式）
    table_str = comparison_df.to_string(index=False)
    
    report = f"""
# 多站点光伏发电功率预测性能综合对比报告

## 对比表格

```
{table_str}
```

## 性能排名

### RMSE排名（越小越好）
"""
    
    # RMSE排名
    rmse_ranking = sorted(all_metrics.items(), key=lambda x: x[1]['RMSE'])
    for i, (station, metrics) in enumerate(rmse_ranking, 1):
        report += f"{i}. {station}: {metrics['RMSE']:.6f}\n"
    
    report += "\n### 相关系数排名（越大越好）\n"
    
    # 相关系数排名
    corr_ranking = sorted(all_metrics.items(), key=lambda x: x[1]['Correlation'], reverse=True)
    for i, (station, metrics) in enumerate(corr_ranking, 1):
        report += f"{i}. {station}: {metrics['Correlation']:.4f}\n"
    
    report += "\n### 合格率排名（越大越好）\n"
    
    # 合格率排名
    qr_ranking = sorted(all_metrics.items(), key=lambda x: x[1]['Qualification_Rate'], reverse=True)
    for i, (station, metrics) in enumerate(qr_ranking, 1):
        report += f"{i}. {station}: {metrics['Qualification_Rate']:.2f}%\n"
    
    report += f"""
## 总体评价

- **最佳RMSE**: {rmse_ranking[0][0]} ({rmse_ranking[0][1]['RMSE']:.6f})
- **最佳相关性**: {corr_ranking[0][0]} ({corr_ranking[0][1]['Correlation']:.4f})
- **最佳合格率**: {qr_ranking[0][0]} ({qr_ranking[0][1]['Qualification_Rate']:.2f}%)

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存综合报告
    with open('results/comprehensive_evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 综合对比报告已保存: results/comprehensive_evaluation_report.md")

if __name__ == "__main__":
    # 评估所有站点的预测结果
    metrics = evaluate_station_predictions() 