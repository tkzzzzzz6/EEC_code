# 光伏发电理论vs实际性能主分析脚本
"""
问题1解决方案：光伏电站发电行为深入理解

通过太阳辐照理论计算，得到理想状态下电站"应该"发出的电量（理论功率），
然后与实际发电量进行对比，揭示光伏发电功率的特点：
1. 季节变化 (长周期): 太阳高度角随季节变化，影响辐照强度
2. 日内波动 (短周期): 太阳升落、云层遮挡、天气等因素影响

通过分析实际功率与理论功率的偏差，了解云量、灰尘、设备效率等
非地理和理论因素对发电的影响，全面掌握光伏电站的发电"脾气"。
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from solar_theoretical_model import PVPerformanceAnalyzer, SolarTheoreticalModel
from performance_visualization import PerformanceVisualizer

warnings.filterwarnings('ignore')

class ComprehensivePVAnalysis:
    """综合光伏性能分析系统"""
    
    def __init__(self, data_dir: str = "../PVODdatasets_v1.0"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化分析器
        self.analyzer = PVPerformanceAnalyzer(data_dir)
        self.visualizer = PerformanceVisualizer()
        
        print("🚀 光伏发电理论vs实际性能综合分析系统初始化完成")
        print(f"📁 数据目录: {self.data_dir}")
        print(f"📁 输出目录: {self.output_dir}")
    
    def analyze_single_station(self, station_id: str, create_visualizations: bool = True):
        """分析单个站点的性能"""
        print(f"\n{'='*60}")
        print(f"🔍 开始分析 {station_id} 站点")
        print(f"{'='*60}")
        
        try:
            # 1. 计算理论性能
            print(f"\n📊 步骤1: 计算 {station_id} 的理论性能...")
            performance_df = self.analyzer.calculate_theoretical_performance(station_id)
            
            # 2. 分析性能模式
            print(f"\n📈 步骤2: 分析 {station_id} 的性能模式...")
            stats = self.analyzer.analyze_performance_patterns(performance_df, station_id)
            
            # 3. 保存数据结果
            csv_path = self.output_dir / f"{station_id}_theoretical_vs_actual.csv"
            performance_df.to_csv(csv_path, index=False)
            print(f"✅ 数据已保存到: {csv_path}")
            
            # 4. 生成可视化分析
            if create_visualizations:
                print(f"\n🎨 步骤3: 生成 {station_id} 的可视化分析...")
                self.visualizer.create_comprehensive_report(performance_df, station_id, stats)
            
            # 5. 打印分析结果
            self.print_analysis_summary(station_id, stats, performance_df)
            
            return performance_df, stats
            
        except Exception as e:
            print(f"❌ 分析 {station_id} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def analyze_multiple_stations(self, station_ids: list = None, max_stations: int = 3):
        """分析多个站点的性能"""
        if station_ids is None:
            # 默认选择数据量较大的几个站点
            metadata = self.analyzer.metadata
            station_ids = metadata['Station_ID'].tolist()[:max_stations]
        
        print(f"\n🔄 开始批量分析 {len(station_ids)} 个站点: {station_ids}")
        
        all_results = {}
        
        for station_id in station_ids:
            performance_df, stats = self.analyze_single_station(station_id, create_visualizations=True)
            if performance_df is not None:
                all_results[station_id] = {
                    'data': performance_df,
                    'stats': stats
                }
        
        # 生成对比分析
        if len(all_results) > 1:
            self.create_comparative_analysis(all_results)
        
        return all_results
    
    def create_comparative_analysis(self, all_results: dict):
        """创建多站点对比分析"""
        print(f"\n📊 生成多站点对比分析...")
        
        # 收集所有站点的统计数据
        comparison_data = []
        
        for station_id, result in all_results.items():
            stats = result['stats']
            comparison_data.append({
                'Station_ID': station_id,
                '平均实际功率(MW)': stats.get('mean_actual_power', 0),
                '平均理论功率(MW)': stats.get('mean_theoretical_power', 0),
                '平均性能比': stats.get('mean_performance_ratio', 0),
                '性能比标准差': stats.get('std_performance_ratio', 0),
                '最大性能比': stats.get('max_performance_ratio', 0),
                '最小性能比': stats.get('min_performance_ratio', 0),
                '白天数据点数': stats.get('daytime_records', 0)
            })
        
        # 保存对比结果
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = self.output_dir / "stations_performance_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"✅ 多站点对比分析已保存到: {comparison_path}")
        print(f"\n📈 多站点性能对比摘要:")
        print(comparison_df.round(3).to_string(index=False))
        
        return comparison_df
    
    def print_analysis_summary(self, station_id: str, stats: dict, performance_df: pd.DataFrame):
        """打印分析摘要"""
        print(f"\n📋 {station_id} 性能分析摘要")
        print(f"{'='*50}")
        
        # 基本统计
        print(f"📊 基本统计:")
        print(f"  总数据记录数: {stats.get('total_records', 0):,}")
        print(f"  白天数据记录数: {stats.get('daytime_records', 0):,}")
        print(f"  平均实际功率: {stats.get('mean_actual_power', 0):.3f} MW")
        print(f"  平均理论功率: {stats.get('mean_theoretical_power', 0):.3f} MW")
        print(f"  平均性能比: {stats.get('mean_performance_ratio', 0):.3f}")
        print(f"  性能比标准差: {stats.get('std_performance_ratio', 0):.3f}")
        print(f"  最大性能比: {stats.get('max_performance_ratio', 0):.3f}")
        print(f"  最小性能比: {stats.get('min_performance_ratio', 0):.3f}")
        
        # 功率损失分析
        power_loss_mw = stats.get('total_power_loss_mw', 0)
        power_loss_percent = stats.get('power_loss_percent', 0)
        print(f"  总功率损失: {power_loss_mw:.2f} MW ({power_loss_percent:.1f}%)")
        
        # 性能评估
        mean_pr = stats.get('mean_performance_ratio', 0)
        if mean_pr > 0.8:
            performance_level = "优秀 🌟"
        elif mean_pr > 0.7:
            performance_level = "良好 👍"
        elif mean_pr > 0.6:
            performance_level = "一般 ⚠️"
        else:
            performance_level = "较差 ❌"
        
        print(f"\n🎯 性能评估: {performance_level}")
        
        # 季节性分析摘要
        if 'seasonal_analysis' in stats:
            print(f"\n🌍 季节性分析:")
            seasonal_data = stats['seasonal_analysis']
            for season in ['春季', '夏季', '秋季', '冬季']:
                if season in seasonal_data.index:
                    pr_mean = seasonal_data.loc[season, ('performance_ratio', 'mean')]
                    print(f"  {season}: 平均性能比 {pr_mean:.3f}")
        
        # 关键发现
        print(f"\n🔍 关键发现:")
        
        # 分析功率损失 - 只基于白昼数据
        daytime_df = performance_df[
            (performance_df['solar_elevation'] > 0) & 
            (performance_df['theoretical_power'] > 0) & 
            (performance_df['power'] >= 0)
        ]
        
        if len(daytime_df) > 0:
            # 分析主要损失原因
            low_performance_data = daytime_df[daytime_df['performance_ratio'] < 0.5]
            if len(low_performance_data) > 0:
                low_performance_ratio = len(low_performance_data) / len(daytime_df) * 100
                print(f"  低性能时段占比: {low_performance_ratio:.1f}% (性能比<0.5)")
            
            # 分析最佳性能时段
            high_performance_data = daytime_df[daytime_df['performance_ratio'] > 0.9]
            if len(high_performance_data) > 0:
                high_performance_ratio = len(high_performance_data) / len(daytime_df) * 100
                print(f"  高性能时段占比: {high_performance_ratio:.1f}% (性能比>0.9)")
            
            # 分析时区修正效果
            print(f"  时区修正: UTC时间已转换为北京时间进行太阳位置计算")
            print(f"  数据过滤: 仅分析白昼时段（太阳高度角>0）的数据")
        
        print(f"{'='*50}")
    
    def generate_comprehensive_report(self, station_ids: list = None):
        """生成综合分析报告"""
        print(f"\n📝 生成综合分析报告...")
        
        # 分析多个站点
        results = self.analyze_multiple_stations(station_ids)
        
        # 创建报告文件
        report_path = self.output_dir / "comprehensive_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 光伏发电理论vs实际性能综合分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 分析概述\n\n")
            f.write("本报告通过太阳辐照理论模型计算光伏电站的理论发电功率，")
            f.write("并与实际发电功率进行对比分析，揭示光伏发电的季节性变化和日内波动特征。\n\n")
            
            f.write("## 主要发现\n\n")
            f.write("### 1. 季节变化特征\n")
            f.write("- 太阳高度角随季节变化，直接影响辐照强度和发电功率\n")
            f.write("- 夏季发电性能通常优于冬季\n")
            f.write("- 春秋季节表现出过渡性特征\n\n")
            
            f.write("### 2. 日内波动特征\n")
            f.write("- 发电功率呈现明显的日内变化模式\n")
            f.write("- 中午时段发电效率最高\n")
            f.write("- 早晚时段受太阳高度角影响较大\n\n")
            
            f.write("### 3. 环境因素影响\n")
            f.write("- 云层遮挡、天气变化对实际发电功率影响显著\n")
            f.write("- 温度、湿度、风速等气象因素对性能有不同程度影响\n")
            f.write("- 设备效率、灰尘积累等因素造成额外功率损失\n\n")
            
            if results:
                f.write("## 站点分析结果\n\n")
                for station_id, result in results.items():
                    stats = result['stats']
                    f.write(f"### {station_id}\n")
                    f.write(f"- 平均性能比: {stats.get('mean_performance_ratio', 0):.3f}\n")
                    f.write(f"- 平均实际功率: {stats.get('mean_actual_power', 0):.3f} MW\n")
                    f.write(f"- 平均理论功率: {stats.get('mean_theoretical_power', 0):.3f} MW\n\n")
            
            f.write("## 结论与建议\n\n")
            f.write("1. **性能监控**: 建立实时性能监控系统，及时发现异常\n")
            f.write("2. **维护优化**: 定期清洁光伏板，优化设备运行效率\n")
            f.write("3. **预测模型**: 基于理论模型建立发电功率预测系统\n")
            f.write("4. **运维策略**: 根据季节性和日内变化特征制定运维计划\n\n")
        
        print(f"✅ 综合分析报告已保存到: {report_path}")
        
        return report_path


def main():
    """主函数"""
    print("🌞 光伏发电理论vs实际性能分析系统")
    print("=" * 60)
    print("问题1解决方案：深入理解光伏电站的发电行为")
    print("通过太阳辐照理论计算与实际功率对比，揭示发电特性")
    print("=" * 60)
    
    try:
        # 初始化分析系统
        analysis_system = ComprehensivePVAnalysis()
        
        # 选择要分析的站点 (可以修改这里选择不同的站点)
        target_stations = ["station01", "station04", "station09"]  # 选择3个代表性站点
        
        print(f"\n🎯 目标分析站点: {target_stations}")
        
        # 生成综合分析报告
        report_path = analysis_system.generate_comprehensive_report(target_stations)
        
        print(f"\n🎉 分析完成！")
        print(f"📁 所有结果已保存到: p1/results/")
        print(f"📊 数据文件: *_theoretical_vs_actual.csv")
        print(f"🎨 可视化图表: figures/")
        print(f"📝 综合报告: {report_path}")
        
        print(f"\n💡 使用建议:")
        print(f"1. 查看CSV文件了解详细数据")
        print(f"2. 查看figures目录中的可视化图表")
        print(f"3. 阅读综合分析报告获取总结")
        
    except Exception as e:
        print(f"❌ 系统运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 