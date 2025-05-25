# NWP信息融入的光伏发电功率预测评价指标计算 + 可视化
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 简化的中文字体设置
import matplotlib as mpl
font_path = 'C:/Windows/Fonts/simhei.ttf'
mpl.font_manager.fontManager.addfont(font_path)  
mpl.rc('font', family='simhei')
plt.rcParams['axes.unicode_minus'] = False

sns.set_palette("husl")
plt.ioff()

class NWPPowerPredictionEvaluator:
    """NWP信息融入的光伏发电功率预测评价指标计算器"""
    
    def __init__(self, capacity):
        """
        初始化评价器
        
        Args:
            capacity: 开机容量 (MW)
        """
        self.capacity = capacity
    
    def ensure_chinese_font(self):
        """确保中文字体设置正确应用"""
        mpl.rc('font', family='simhei')
        plt.rcParams['axes.unicode_minus'] = False
    
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
        
        # 7. 额外的NWP评价指标
        # 平均绝对百分比误差 (MAPE)
        mape = np.mean(np.abs(normalized_error)) * 100
        
        # 标准化均方根误差 (NRMSE)
        nrmse = rmse / (np.max(normalized_actual) - np.min(normalized_actual)) if np.max(normalized_actual) != np.min(normalized_actual) else 0
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'ME': me,
            'Correlation': correlation,
            'Accuracy': accuracy,
            'Qualification_Rate': qualification_rate,
            'MAPE': mape,
            'NRMSE': nrmse,
            'Sample_Count': n
        }
    
    def compare_models(self, actual_power, model_predictions, model_names):
        """
        对比多个模型的性能
        
        Args:
            actual_power: 实际功率数组
            model_predictions: 字典，包含各模型的预测结果
            model_names: 模型名称列表
            
        Returns:
            dict: 各模型的评价指标
        """
        all_metrics = {}
        
        for model_name in model_names:
            if model_name in model_predictions:
                metrics = self.calculate_metrics(actual_power, model_predictions[model_name])
                all_metrics[model_name] = metrics
        
        return all_metrics
    
    def analyze_nwp_effectiveness(self, all_metrics):
        """
        分析NWP信息的有效性
        
        Args:
            all_metrics: 包含各模型评价指标的字典
            
        Returns:
            dict: NWP有效性分析结果
        """
        if 'basic' not in all_metrics or 'nwp_enhanced' not in all_metrics:
            return {}
        
        basic_metrics = all_metrics['basic']
        nwp_metrics = all_metrics['nwp_enhanced']
        
        # 计算改善程度
        rmse_improvement = (basic_metrics['RMSE'] - nwp_metrics['RMSE']) / basic_metrics['RMSE'] * 100
        mae_improvement = (basic_metrics['MAE'] - nwp_metrics['MAE']) / basic_metrics['MAE'] * 100
        accuracy_improvement = nwp_metrics['Accuracy'] - basic_metrics['Accuracy']
        correlation_improvement = nwp_metrics['Correlation'] - basic_metrics['Correlation']
        qualification_improvement = nwp_metrics['Qualification_Rate'] - basic_metrics['Qualification_Rate']
        
        # 判断有效性等级
        if rmse_improvement > 5 and accuracy_improvement > 2:
            effectiveness_level = "显著有效"
        elif rmse_improvement > 2 and accuracy_improvement > 1:
            effectiveness_level = "有效"
        elif rmse_improvement > 0:
            effectiveness_level = "轻微有效"
        else:
            effectiveness_level = "无效或负面影响"
        
        return {
            'rmse_improvement': rmse_improvement,
            'mae_improvement': mae_improvement,
            'accuracy_improvement': accuracy_improvement,
            'correlation_improvement': correlation_improvement,
            'qualification_improvement': qualification_improvement,
            'effectiveness_level': effectiveness_level
        }
    
    def create_nwp_comparison_visualization(self, actual, model_predictions, station_id, save_dir="results/figures"):
        """创建NWP对比可视化图表"""
        self.ensure_chinese_font()
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 只使用白天时段的数据
        daytime_mask = actual > 0
        actual_day = actual[daytime_mask]
        
        if len(actual_day) == 0:
            print(f"⚠️ {station_id} 没有白天时段数据，跳过可视化")
            return
        
        # 创建综合对比图表
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'{station_id} NWP信息融入效果分析', fontsize=16, fontweight='bold')
        
        # 1. 时间序列对比
        ax1 = axes[0, 0]
        time_index = range(len(actual))
        ax1.plot(time_index, actual, label='实际功率', alpha=0.8, linewidth=1.5, color='black')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, predictions) in enumerate(model_predictions.items()):
            if i < len(colors):
                ax1.plot(time_index, predictions, label=f'{model_name}', 
                        alpha=0.7, linewidth=1.2, color=colors[i])
        
        ax1.set_title('时间序列对比')
        ax1.set_xlabel('时间点')
        ax1.set_ylabel('功率 (MW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 散点图对比（基础模型 vs NWP增强模型）
        ax2 = axes[0, 1]
        if 'basic' in model_predictions and 'nwp_enhanced' in model_predictions:
            basic_day = model_predictions['basic'][daytime_mask]
            nwp_day = model_predictions['nwp_enhanced'][daytime_mask]
            
            ax2.scatter(actual_day, basic_day, alpha=0.6, s=20, color='red', label='基础模型')
            ax2.scatter(actual_day, nwp_day, alpha=0.6, s=20, color='blue', label='NWP增强模型')
            
            max_val = max(actual_day.max(), basic_day.max(), nwp_day.max())
            ax2.plot([0, max_val], [0, max_val], 'k--', label='理想预测线')
            ax2.set_title('散点图对比 (仅白天时段)')
            ax2.set_xlabel('实际功率 (MW)')
            ax2.set_ylabel('预测功率 (MW)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 误差分布对比
        ax3 = axes[1, 0]
        if 'basic' in model_predictions and 'nwp_enhanced' in model_predictions:
            basic_errors = model_predictions['basic'][daytime_mask] - actual_day
            nwp_errors = model_predictions['nwp_enhanced'][daytime_mask] - actual_day
            
            ax3.hist(basic_errors, bins=30, alpha=0.7, color='red', label='基础模型误差', density=True)
            ax3.hist(nwp_errors, bins=30, alpha=0.7, color='blue', label='NWP增强模型误差', density=True)
            ax3.axvline(basic_errors.mean(), color='red', linestyle='--', alpha=0.8)
            ax3.axvline(nwp_errors.mean(), color='blue', linestyle='--', alpha=0.8)
            ax3.set_title('误差分布对比')
            ax3.set_xlabel('预测误差 (MW)')
            ax3.set_ylabel('密度')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 模型性能雷达图
        ax4 = axes[1, 1]
        all_metrics = self.compare_models(actual, model_predictions, list(model_predictions.keys()))
        
        if len(all_metrics) >= 2:
            # 选择主要的两个模型进行雷达图对比
            models_to_compare = ['basic', 'nwp_enhanced'] if 'basic' in all_metrics and 'nwp_enhanced' in all_metrics else list(all_metrics.keys())[:2]
            
            categories = ['准确率', '合格率', '相关系数', 'RMSE(反)', 'MAE(反)']
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))
            
            ax4 = plt.subplot(3, 2, 4, projection='polar')
            
            for model_name in models_to_compare:
                if model_name in all_metrics:
                    metrics = all_metrics[model_name]
                    values = [
                        metrics['Accuracy'],
                        metrics['Qualification_Rate'],
                        metrics['Correlation'] * 100,
                        (1 - metrics['RMSE']) * 100,
                        (1 - metrics['MAE']) * 100
                    ]
                    values += values[:1]
                    
                    ax4.plot(angles, values, 'o-', linewidth=2, label=model_name)
                    ax4.fill(angles, values, alpha=0.25)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories)
            ax4.set_ylim(0, 100)
            ax4.set_title('模型性能雷达图', pad=20)
            ax4.legend()
            ax4.grid(True)
        
        # 5. 改善效果分析
        ax5 = axes[2, 0]
        if len(all_metrics) >= 2:
            effectiveness_analysis = self.analyze_nwp_effectiveness(all_metrics)
            
            if effectiveness_analysis:
                improvements = [
                    effectiveness_analysis['rmse_improvement'],
                    effectiveness_analysis['mae_improvement'],
                    effectiveness_analysis['accuracy_improvement'],
                    effectiveness_analysis['correlation_improvement'] * 100,
                    effectiveness_analysis['qualification_improvement']
                ]
                
                improvement_names = ['RMSE改善(%)', 'MAE改善(%)', '准确率提升', '相关系数提升(%)', '合格率提升']
                
                colors_bar = ['red' if x < 0 else 'green' for x in improvements]
                bars = ax5.bar(improvement_names, improvements, color=colors_bar, alpha=0.7)
                
                ax5.set_title('NWP信息改善效果')
                ax5.set_ylabel('改善程度')
                ax5.grid(True, alpha=0.3)
                ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                # 添加数值标签
                for bar, value in zip(bars, improvements):
                    ax5.text(bar.get_x() + bar.get_width()/2, 
                            bar.get_height() + (0.1 if value >= 0 else -0.3),
                            f'{value:.2f}', ha='center', va='bottom' if value >= 0 else 'top')
                
                plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 6. 累积误差对比
        ax6 = axes[2, 1]
        if 'basic' in model_predictions and 'nwp_enhanced' in model_predictions:
            basic_errors = model_predictions['basic'][daytime_mask] - actual_day
            nwp_errors = model_predictions['nwp_enhanced'][daytime_mask] - actual_day
            
            cumulative_basic = np.cumsum(np.abs(basic_errors))
            cumulative_nwp = np.cumsum(np.abs(nwp_errors))
            
            day_time_index = range(len(basic_errors))
            ax6.plot(day_time_index, cumulative_basic, label='基础模型累积误差', color='red', linewidth=2)
            ax6.plot(day_time_index, cumulative_nwp, label='NWP增强模型累积误差', color='blue', linewidth=2)
            
            ax6.set_title('累积绝对误差对比')
            ax6.set_xlabel('白天时间点')
            ax6.set_ylabel('累积绝对误差 (MW)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{station_id}_nwp_comprehensive_evaluation.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {station_id} NWP综合评价图表已生成")
    
    def create_scenario_effectiveness_chart(self, scenarios, station_id, save_dir="results/figures"):
        """创建场景有效性分析图表"""
        self.ensure_chinese_font()
        
        if not scenarios:
            return
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{station_id} NWP信息场景有效性分析', fontsize=16, fontweight='bold')
        
        scenario_names = [scenarios[k]['description'] for k in scenarios.keys()]
        improvements = [scenarios[k]['improvement'] for k in scenarios.keys()]
        sample_counts = [scenarios[k]['sample_count'] for k in scenarios.keys()]
        
        # 1. 场景改善效果柱状图
        ax1 = axes[0, 0]
        colors = ['green' if x > 0 else 'red' for x in improvements]
        bars = ax1.bar(range(len(scenario_names)), improvements, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(scenario_names)))
        ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax1.set_ylabel('误差改善 (MW)')
        ax1.set_title('各场景NWP改善效果')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 添加数值标签
        for bar, value in zip(bars, improvements):
            ax1.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.01 if value >= 0 else -0.02),
                    f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)
        
        # 2. 样本数量分布
        ax2 = axes[0, 1]
        ax2.bar(range(len(scenario_names)), sample_counts, alpha=0.7, color='skyblue')
        ax2.set_xticks(range(len(scenario_names)))
        ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax2.set_ylabel('样本数量')
        ax2.set_title('各场景样本数量')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, count in enumerate(sample_counts):
            ax2.text(i, count + max(sample_counts) * 0.01, str(count), 
                    ha='center', va='bottom', fontsize=9)
        
        # 3. 改善效果vs样本数量散点图
        ax3 = axes[1, 0]
        scatter = ax3.scatter(sample_counts, improvements, s=100, alpha=0.7, c=improvements, cmap='RdYlGn')
        ax3.set_xlabel('样本数量')
        ax3.set_ylabel('误差改善 (MW)')
        ax3.set_title('改善效果 vs 样本数量')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 添加场景标签
        for i, name in enumerate(scenario_names):
            ax3.annotate(name, (sample_counts[i], improvements[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=ax3, label='改善程度')
        
        # 4. 有效性分类饼图
        ax4 = axes[1, 1]
        positive_count = sum(1 for x in improvements if x > 0)
        negative_count = sum(1 for x in improvements if x <= 0)
        
        if positive_count + negative_count > 0:
            labels = ['有效场景', '无效场景']
            sizes = [positive_count, negative_count]
            colors_pie = ['lightgreen', 'lightcoral']
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie, 
                                              autopct='%1.1f%%', startangle=90)
            ax4.set_title('场景有效性分布')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{station_id}_scenario_effectiveness.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {station_id} 场景有效性分析图表已生成")
    
    def create_multi_station_nwp_comparison(self, all_station_metrics, save_dir="results/figures"):
        """创建多站点NWP效果对比图表"""
        self.ensure_chinese_font()
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        stations = list(all_station_metrics.keys())
        
        # 提取各站点的基础模型和NWP增强模型指标
        basic_metrics = []
        nwp_metrics = []
        effectiveness_levels = []
        
        for station in stations:
            if 'basic' in all_station_metrics[station] and 'nwp_enhanced' in all_station_metrics[station]:
                basic_metrics.append(all_station_metrics[station]['basic'])
                nwp_metrics.append(all_station_metrics[station]['nwp_enhanced'])
                
                # 计算有效性等级
                effectiveness = self.analyze_nwp_effectiveness({
                    'basic': all_station_metrics[station]['basic'],
                    'nwp_enhanced': all_station_metrics[station]['nwp_enhanced']
                })
                effectiveness_levels.append(effectiveness.get('effectiveness_level', '未知'))
        
        if not basic_metrics:
            print("⚠️ 没有足够的数据进行多站点对比")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('多站点NWP信息融入效果对比分析', fontsize=16, fontweight='bold')
        
        # 1. RMSE对比
        ax1 = axes[0, 0]
        basic_rmse = [m['RMSE'] for m in basic_metrics]
        nwp_rmse = [m['RMSE'] for m in nwp_metrics]
        
        x = np.arange(len(stations))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, basic_rmse, width, label='基础模型', alpha=0.7, color='red')
        bars2 = ax1.bar(x + width/2, nwp_rmse, width, label='NWP增强模型', alpha=0.7, color='blue')
        
        ax1.set_xlabel('站点')
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(stations)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率对比
        ax2 = axes[0, 1]
        basic_acc = [m['Accuracy'] for m in basic_metrics]
        nwp_acc = [m['Accuracy'] for m in nwp_metrics]
        
        bars3 = ax2.bar(x - width/2, basic_acc, width, label='基础模型', alpha=0.7, color='red')
        bars4 = ax2.bar(x + width/2, nwp_acc, width, label='NWP增强模型', alpha=0.7, color='blue')
        
        ax2.set_xlabel('站点')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title('准确率对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(stations)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 改善程度分析
        ax3 = axes[0, 2]
        rmse_improvements = [(basic_rmse[i] - nwp_rmse[i]) / basic_rmse[i] * 100 for i in range(len(stations))]
        acc_improvements = [nwp_acc[i] - basic_acc[i] for i in range(len(stations))]
        
        colors_improve = ['green' if x > 0 else 'red' for x in rmse_improvements]
        bars5 = ax3.bar(stations, rmse_improvements, alpha=0.7, color=colors_improve)
        
        ax3.set_xlabel('站点')
        ax3.set_ylabel('RMSE改善 (%)')
        ax3.set_title('RMSE改善程度')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 添加数值标签
        for bar, value in zip(bars5, rmse_improvements):
            ax3.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.5 if value >= 0 else -1),
                    f'{value:.1f}%', ha='center', va='bottom' if value >= 0 else 'top')
        
        # 4. 相关系数对比
        ax4 = axes[1, 0]
        basic_corr = [m['Correlation'] for m in basic_metrics]
        nwp_corr = [m['Correlation'] for m in nwp_metrics]
        
        bars6 = ax4.bar(x - width/2, basic_corr, width, label='基础模型', alpha=0.7, color='red')
        bars7 = ax4.bar(x + width/2, nwp_corr, width, label='NWP增强模型', alpha=0.7, color='blue')
        
        ax4.set_xlabel('站点')
        ax4.set_ylabel('相关系数')
        ax4.set_title('相关系数对比')
        ax4.set_xticks(x)
        ax4.set_xticklabels(stations)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 有效性等级分布
        ax5 = axes[1, 1]
        effectiveness_counts = {}
        for level in effectiveness_levels:
            effectiveness_counts[level] = effectiveness_counts.get(level, 0) + 1
        
        if effectiveness_counts:
            labels = list(effectiveness_counts.keys())
            sizes = list(effectiveness_counts.values())
            colors_pie = ['lightgreen', 'yellow', 'orange', 'lightcoral'][:len(labels)]
            
            wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=colors_pie, 
                                              autopct='%1.1f%%', startangle=90)
            ax5.set_title('NWP有效性等级分布')
        
        # 6. 综合性能雷达图
        ax6 = axes[1, 2]
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        
        # 计算平均性能
        avg_basic_metrics = {
            'RMSE': np.mean(basic_rmse),
            'Accuracy': np.mean(basic_acc),
            'Correlation': np.mean(basic_corr),
            'MAE': np.mean([m['MAE'] for m in basic_metrics]),
            'Qualification_Rate': np.mean([m['Qualification_Rate'] for m in basic_metrics])
        }
        
        avg_nwp_metrics = {
            'RMSE': np.mean(nwp_rmse),
            'Accuracy': np.mean(nwp_acc),
            'Correlation': np.mean(nwp_corr),
            'MAE': np.mean([m['MAE'] for m in nwp_metrics]),
            'Qualification_Rate': np.mean([m['Qualification_Rate'] for m in nwp_metrics])
        }
        
        categories = ['准确率', '合格率', '相关系数', 'RMSE(反)', 'MAE(反)']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        basic_values = [
            avg_basic_metrics['Accuracy'],
            avg_basic_metrics['Qualification_Rate'],
            avg_basic_metrics['Correlation'] * 100,
            (1 - avg_basic_metrics['RMSE']) * 100,
            (1 - avg_basic_metrics['MAE']) * 100
        ]
        basic_values += basic_values[:1]
        
        nwp_values = [
            avg_nwp_metrics['Accuracy'],
            avg_nwp_metrics['Qualification_Rate'],
            avg_nwp_metrics['Correlation'] * 100,
            (1 - avg_nwp_metrics['RMSE']) * 100,
            (1 - avg_nwp_metrics['MAE']) * 100
        ]
        nwp_values += nwp_values[:1]
        
        ax6.plot(angles, basic_values, 'o-', linewidth=2, label='基础模型平均', color='red')
        ax6.fill(angles, basic_values, alpha=0.25, color='red')
        ax6.plot(angles, nwp_values, 'o-', linewidth=2, label='NWP增强模型平均', color='blue')
        ax6.fill(angles, nwp_values, alpha=0.25, color='blue')
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 100)
        ax6.set_title('平均性能雷达图', pad=20)
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'multi_station_nwp_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 多站点NWP对比分析图表已生成")

def evaluate_nwp_predictions():
    """评估NWP预测结果"""
    print("🔍 开始评估NWP预测结果...")
    
    stations = ['station00', 'station04', 'station05', 'station09']
    all_station_metrics = {}
    
    # 开机容量设定
    capacities = {
        'station00': 6.628,
        'station04': 32.122,
        'station05': 42.142,
        'station09': 14.454
    }
    
    for station in stations:
        try:
            print(f"\n{'='*20} 评估 {station} {'='*20}")
            
            # 读取NWP预测结果
            results_file = f'results/{station}_nwp_prediction_results.csv'
            if not Path(results_file).exists():
                print(f"❌ 找不到 {station} 的NWP预测结果文件")
                continue
            
            results_df = pd.read_csv(results_file)
            actual_power = results_df['actual_power'].values
            
            # 提取各模型的预测结果
            model_predictions = {}
            for col in results_df.columns:
                if col.endswith('_prediction'):
                    model_name = col.replace('_prediction', '')
                    model_predictions[model_name] = results_df[col].values
            
            if not model_predictions:
                print(f"❌ {station} 没有找到模型预测结果")
                continue
            
            # 创建评价器
            evaluator = NWPPowerPredictionEvaluator(capacities[station])
            
            # 计算各模型的评价指标
            all_metrics = evaluator.compare_models(actual_power, model_predictions, list(model_predictions.keys()))
            all_station_metrics[station] = all_metrics
            
            # 分析NWP有效性
            effectiveness_analysis = evaluator.analyze_nwp_effectiveness(all_metrics)
            
            # 打印评价结果
            print(f"\n📊 {station} 模型性能对比")
            print("="*60)
            for model_name, metrics in all_metrics.items():
                print(f"\n{model_name.upper()} 模型:")
                print(f"  RMSE: {metrics['RMSE']:.6f}")
                print(f"  MAE: {metrics['MAE']:.6f}")
                print(f"  相关系数: {metrics['Correlation']:.4f}")
                print(f"  准确率: {metrics['Accuracy']:.2f}%")
                print(f"  合格率: {metrics['Qualification_Rate']:.2f}%")
            
            if effectiveness_analysis:
                print(f"\n🎯 NWP有效性分析:")
                print(f"  RMSE改善: {effectiveness_analysis['rmse_improvement']:.2f}%")
                print(f"  准确率提升: {effectiveness_analysis['accuracy_improvement']:.2f}个百分点")
                print(f"  有效性等级: {effectiveness_analysis['effectiveness_level']}")
            
            # 创建可视化
            evaluator.create_nwp_comparison_visualization(
                actual_power, model_predictions, station
            )
            
            # 保存详细评价报告
            save_nwp_evaluation_report(station, all_metrics, effectiveness_analysis, 
                                     capacities[station], actual_power, model_predictions)
            
        except Exception as e:
            print(f"❌ 评估 {station} 时出错: {str(e)}")
            continue
    
    # 创建多站点对比
    if all_station_metrics:
        evaluator = NWPPowerPredictionEvaluator(1.0)  # 临时创建用于多站点对比
        evaluator.create_multi_station_nwp_comparison(all_station_metrics)
        create_nwp_comprehensive_report(all_station_metrics)
    
    print(f"\n🎉 NWP预测结果评估完成！")
    return all_station_metrics

def save_nwp_evaluation_report(station_id, all_metrics, effectiveness_analysis, capacity, actual_power, model_predictions):
    """保存NWP评价报告"""
    
    report = f"""
# {station_id} NWP信息融入光伏发电功率预测评价报告

## 基本信息
- **站点**: {station_id}
- **开机容量**: {capacity:.3f} MW
- **评估模型数**: {len(all_metrics)}
- **实际功率均值**: {actual_power[actual_power > 0].mean():.3f} MW
- **实际功率最大值**: {actual_power.max():.3f} MW

## 模型性能对比

"""
    
    for model_name, metrics in all_metrics.items():
        report += f"""
### {model_name.upper()} 模型
- **RMSE**: {metrics['RMSE']:.6f}
- **MAE**: {metrics['MAE']:.6f}
- **平均误差 (ME)**: {metrics['ME']:.6f}
- **相关系数**: {metrics['Correlation']:.6f}
- **准确率**: {metrics['Accuracy']:.2f}%
- **合格率**: {metrics['Qualification_Rate']:.2f}%
- **MAPE**: {metrics['MAPE']:.2f}%
- **样本数**: {metrics['Sample_Count']}

"""
    
    if effectiveness_analysis:
        report += f"""
## NWP信息有效性分析

### 改善效果
- **RMSE改善**: {effectiveness_analysis['rmse_improvement']:.2f}%
- **MAE改善**: {effectiveness_analysis['mae_improvement']:.2f}%
- **准确率提升**: {effectiveness_analysis['accuracy_improvement']:.2f}个百分点
- **相关系数提升**: {effectiveness_analysis['correlation_improvement']:.4f}
- **合格率提升**: {effectiveness_analysis['qualification_improvement']:.2f}个百分点

### 有效性评价
- **等级**: {effectiveness_analysis['effectiveness_level']}

"""
        
        # 根据有效性等级给出建议
        if effectiveness_analysis['effectiveness_level'] == "显著有效":
            recommendation = "强烈建议在实际应用中采用NWP增强模型，可显著提高预测精度。"
        elif effectiveness_analysis['effectiveness_level'] == "有效":
            recommendation = "建议在实际应用中采用NWP增强模型，可有效提高预测精度。"
        elif effectiveness_analysis['effectiveness_level'] == "轻微有效":
            recommendation = "可考虑在特定场景下使用NWP信息，但改善效果有限。"
        else:
            recommendation = "不建议使用当前的NWP信息，可能需要优化数据质量或特征工程方法。"
        
        report += f"""
### 应用建议
{recommendation}

"""
    
    report += f"""
## 模型排名

### 按RMSE排序（越小越好）
"""
    
    sorted_by_rmse = sorted(all_metrics.items(), key=lambda x: x[1]['RMSE'])
    for i, (model_name, metrics) in enumerate(sorted_by_rmse, 1):
        report += f"{i}. {model_name}: {metrics['RMSE']:.6f}\n"
    
    report += f"""
### 按准确率排序（越大越好）
"""
    
    sorted_by_accuracy = sorted(all_metrics.items(), key=lambda x: x[1]['Accuracy'], reverse=True)
    for i, (model_name, metrics) in enumerate(sorted_by_accuracy, 1):
        report += f"{i}. {model_name}: {metrics['Accuracy']:.2f}%\n"
    
    report += f"""
---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存报告
    with open(f'results/{station_id}_nwp_evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ NWP评价报告已保存: results\\{station_id}_nwp_evaluation_report.md")

def create_nwp_comprehensive_report(all_station_metrics):
    """创建NWP综合对比报告"""
    print(f"\n{'='*80}")
    print("📊 NWP信息融入效果综合对比报告")
    print("="*80)
    
    # 统计有效性
    effectiveness_stats = {}
    improvement_stats = []
    
    for station, metrics in all_station_metrics.items():
        if 'basic' in metrics and 'nwp_enhanced' in metrics:
            evaluator = NWPPowerPredictionEvaluator(1.0)
            effectiveness = evaluator.analyze_nwp_effectiveness(metrics)
            
            if effectiveness:
                level = effectiveness['effectiveness_level']
                effectiveness_stats[level] = effectiveness_stats.get(level, 0) + 1
                improvement_stats.append({
                    'station': station,
                    'rmse_improvement': effectiveness['rmse_improvement'],
                    'accuracy_improvement': effectiveness['accuracy_improvement'],
                    'effectiveness_level': level
                })
                
                print(f"{station}: {level} (RMSE改善: {effectiveness['rmse_improvement']:.2f}%)")
    
    print(f"\n📈 有效性统计:")
    for level, count in effectiveness_stats.items():
        print(f"  {level}: {count}个站点")
    
    # 计算平均改善效果
    if improvement_stats:
        avg_rmse_improvement = np.mean([s['rmse_improvement'] for s in improvement_stats])
        avg_accuracy_improvement = np.mean([s['accuracy_improvement'] for s in improvement_stats])
        
        print(f"\n📊 平均改善效果:")
        print(f"  平均RMSE改善: {avg_rmse_improvement:.2f}%")
        print(f"  平均准确率提升: {avg_accuracy_improvement:.2f}个百分点")
    
    # 保存综合报告
    report = f"""
# NWP信息融入光伏发电功率预测综合分析报告

## 分析概述
本报告全面分析了NWP（数值天气预报）信息对光伏发电功率预测精度的影响，包括多个模型的对比和不同场景下的有效性分析。

## 站点分析结果

"""
    
    for station, metrics in all_station_metrics.items():
        report += f"""
### {station}
"""
        
        # 模型性能对比表格
        report += "| 模型 | RMSE | MAE | 相关系数 | 准确率(%) | 合格率(%) |\n"
        report += "|------|------|-----|----------|-----------|----------|\n"
        
        for model_name, model_metrics in metrics.items():
            report += f"| {model_name} | {model_metrics['RMSE']:.6f} | {model_metrics['MAE']:.6f} | {model_metrics['Correlation']:.4f} | {model_metrics['Accuracy']:.2f} | {model_metrics['Qualification_Rate']:.2f} |\n"
        
        # NWP有效性分析
        if 'basic' in metrics and 'nwp_enhanced' in metrics:
            evaluator = NWPPowerPredictionEvaluator(1.0)
            effectiveness = evaluator.analyze_nwp_effectiveness(metrics)
            
            if effectiveness:
                report += f"""
**NWP有效性**: {effectiveness['effectiveness_level']}
- RMSE改善: {effectiveness['rmse_improvement']:.2f}%
- 准确率提升: {effectiveness['accuracy_improvement']:.2f}个百分点

"""
    
    report += f"""
## 总体结论

### NWP信息有效性统计
"""
    
    for level, count in effectiveness_stats.items():
        report += f"- **{level}**: {count}个站点\n"
    
    if improvement_stats:
        report += f"""
### 平均改善效果
- **平均RMSE改善**: {avg_rmse_improvement:.2f}%
- **平均准确率提升**: {avg_accuracy_improvement:.2f}个百分点

### 最佳改善站点
"""
        
        # 按RMSE改善排序
        best_stations = sorted(improvement_stats, key=lambda x: x['rmse_improvement'], reverse=True)
        for i, station_data in enumerate(best_stations[:3], 1):
            report += f"{i}. {station_data['station']}: RMSE改善 {station_data['rmse_improvement']:.2f}%\n"
    
    report += f"""
### 应用建议

1. **显著有效站点**: 强烈建议在实际应用中全面采用NWP增强模型
2. **有效站点**: 建议在日常预测中使用NWP信息，特别是在特定天气条件下
3. **轻微有效站点**: 可在关键时段或特殊天气条件下选择性使用NWP信息
4. **无效站点**: 需要进一步优化NWP数据质量、特征工程方法或模型结构

### 技术改进方向

1. **数据质量优化**: 提高NWP数据的时空分辨率和准确性
2. **特征工程改进**: 开发更有效的NWP特征组合和变换方法
3. **模型结构优化**: 探索深度学习等更先进的融合方法
4. **场景化应用**: 针对不同天气条件和时间段优化NWP信息的使用策略

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open('results/nwp_comprehensive_evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ NWP综合评价报告已保存: results/nwp_comprehensive_evaluation_report.md")

if __name__ == "__main__":
    # 评估NWP预测结果
    metrics = evaluate_nwp_predictions() 